"""Sprint 1301 — on-chain settlement go-live preflight.

The TEE Tier-3 → roadmap-F machinery is fully deployed + Foundation-owned + the
canonical config now points at the F bundle (sp1300). But on-chain settlement has
NEVER been activated in production — no funded settler key has committed a real
batch. This module is the read-only/offline go/no-go gate between "the machinery
exists" and "the network settles on-chain", mirroring the Aerodrome pool
``go_live_verification`` harness (PASS/FAIL/WARN/INFO; ``go`` iff zero FAIL).

It checks, without moving any money and without broadcasting any tx:

  provider_address     — the node's on-chain identity resolves
  onchain_settlement   — PRSM_ONCHAIN_SETTLEMENT is enabled
  settler_key          — FTNS_WALLET_PRIVATE_KEY present + CONTROLS provider_address
                         (commitBatch settles to msg.sender → a mismatch pays the
                         wrong party; the client refuses to build, so this FAILs)
  client_build         — build_onchain_settlement_client_or_none yields a
                         WRITE-CAPABLE client (composes the three above end-to-end)
  registry_resolved    — the resolved settlement registry is set (and not the
                         known-retired pre-F registry)
  registry_active      — on-chain ``paused() == false`` (a paused/retired registry
                         rejects every commitBatch)
  attestation_surface  — the registry exposes commitBatchWithAttestation (roadmap F)
  settler_funded       — the settler key holds Base ETH for gas
  escrow_resolved      — the EscrowPool address resolves (funding source)
  attestation_flag     — PRSM_SETTLEMENT_SUPPORTS_ATTESTATION vs the on-chain surface
                         (so the operator knows whether the FIRST batch commits a real
                         attestation or legacy commitBatch(bytes32(0)))

On-chain reads (paused/getCode/balance) go through an injectable ``reader`` so the
whole gate is offline-testable; the default reads Base via web3. Every read is
fail-soft (errors degrade to WARN, never crash the report).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The pre-F registry retired by the sp1300 cutover (paused, Foundation-owned).
# Resolving to it at go-live time means the config was not re-pointed.
_RETIRED_REGISTRY = "0x48fFab641b9D638F312FFA776818756a326F995B"


@dataclass
class PreflightFinding:
    check: str
    status: str  # PASS | FAIL | WARN | INFO
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "status": self.status, "detail": self.detail}


@dataclass
class SettlementGoLiveReport:
    findings: List[PreflightFinding] = field(default_factory=list)

    @property
    def go(self) -> bool:
        """Go-live iff no FAIL findings. WARN/INFO are advisory."""
        return not any(f.status == "FAIL" for f in self.findings)

    def add(self, check: str, status: str, detail: str) -> None:
        self.findings.append(PreflightFinding(check, status, detail))

    def to_dict(self) -> Dict[str, Any]:
        return {"go": self.go, "findings": [f.to_dict() for f in self.findings]}


class _Web3Reader:
    """Default fail-soft on-chain reader (Base via web3). Every method returns
    None on any error so the preflight degrades to advisory WARNs, never crashes."""

    def __init__(self, rpc_url: str) -> None:
        from web3 import Web3
        self._Web3 = Web3
        self._w3 = Web3(Web3.HTTPProvider(rpc_url))

    def paused(self, addr: str) -> Optional[bool]:
        try:
            c = self._w3.eth.contract(
                address=self._Web3.to_checksum_address(addr),
                abi=[{"inputs": [], "name": "paused",
                      "outputs": [{"type": "bool"}],
                      "stateMutability": "view", "type": "function"}],
            )
            return bool(c.functions.paused().call())
        except Exception:  # noqa: BLE001
            return None

    def code_hex(self, addr: str) -> Optional[str]:
        try:
            return self._w3.eth.get_code(
                self._Web3.to_checksum_address(addr)).hex()
        except Exception:  # noqa: BLE001
            return None

    def eth_balance_wei(self, addr: str) -> Optional[int]:
        try:
            return int(self._w3.eth.get_balance(
                self._Web3.to_checksum_address(addr)))
        except Exception:  # noqa: BLE001
            return None


def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def run_settlement_go_live_preflight(
    *,
    env: Optional[dict] = None,
    reader: Optional[Any] = None,
    provider_address: Optional[str] = None,
) -> SettlementGoLiveReport:
    """Run the on-chain settlement go-live preflight. Returns a
    SettlementGoLiveReport whose ``go`` is the operator's launch gate.

    ``env`` defaults to ``os.environ``; ``provider_address`` defaults to
    ``resolve_operator_address()``; ``reader`` defaults to a web3 reader built
    from the resolved RPC URL (injected as a fake in tests)."""
    import os
    environ = env if env is not None else os.environ
    report = SettlementGoLiveReport()

    # ── identity + opt-in + key control (offline) ───────────────────────────
    if provider_address is None:
        from prsm.node.operator_address import resolve_operator_address
        provider_address = resolve_operator_address()
    if provider_address:
        report.add("provider_address", "PASS",
                   f"node on-chain identity {provider_address}")
    else:
        report.add("provider_address", "FAIL",
                   "no provider_address — set PRSM_OPERATOR_ADDRESS or "
                   "FTNS_WALLET_PRIVATE_KEY so the node has an on-chain identity.")

    if _truthy(environ.get("PRSM_ONCHAIN_SETTLEMENT")):
        report.add("onchain_settlement", "PASS",
                   "PRSM_ONCHAIN_SETTLEMENT enabled")
    else:
        report.add("onchain_settlement", "FAIL",
                   "PRSM_ONCHAIN_SETTLEMENT is off — set it to 1 to commit "
                   "batches on-chain (default is local accumulation only).")

    key = (environ.get("FTNS_WALLET_PRIVATE_KEY", "") or "").strip()
    if not key:
        report.add("settler_key", "FAIL",
                   "no FTNS_WALLET_PRIVATE_KEY — without a funded settler key the "
                   "client is VIEW-ONLY (commit/finalize defer); cannot go live.")
        signer = None
    else:
        if not key.startswith("0x"):
            key = "0x" + key
        try:
            from eth_account import Account
            signer = Account.from_key(key).address
        except Exception:  # noqa: BLE001
            signer = None
            report.add("settler_key", "FAIL",
                       "FTNS_WALLET_PRIVATE_KEY is malformed (cannot derive an "
                       "address).")
        if signer is not None:
            prov = (provider_address or "").strip()
            if prov and not prov.startswith("0x"):
                prov = "0x" + prov
            if prov and signer.lower() == prov.lower():
                report.add("settler_key", "PASS",
                           f"settler key controls provider_address ({signer})")
            else:
                report.add("settler_key", "FAIL",
                           f"settler key address {signer} does NOT control "
                           f"provider_address {provider_address} — commitBatch "
                           "settles to msg.sender, so this would pay the wrong "
                           "party (the client refuses to build).")

    # ── end-to-end client build (composes the above; authoritative) ─────────
    try:
        from prsm.settlement.client_wiring import (
            build_onchain_settlement_client_or_none,
        )
        client = build_onchain_settlement_client_or_none(
            provider_address=provider_address, env=environ,
        )
    except Exception as exc:  # noqa: BLE001 — never crash the report
        client = None
        logger.warning("go-live: client build raised: %s", exc)
    write_capable = bool(
        client is not None
        and getattr(getattr(client, "_contract", None), "address", None)
    )
    if write_capable:
        report.add("client_build", "PASS",
                   "settlement client builds WRITE-CAPABLE (funded settler key "
                   "bound to provider_address)")
    elif client is not None:
        report.add("client_build", "FAIL",
                   "settlement client builds but is VIEW-ONLY (no write key) — "
                   "commit/finalize cannot run.")
    else:
        report.add("client_build", "FAIL",
                   "settlement client did not build (see provider/key/registry "
                   "findings) — settlement stays off.")

    # ── resolve the on-chain endpoints (offline) ────────────────────────────
    registry = rpc_url = escrow = None
    try:
        from prsm.config.networks import resolve_endpoints
        endpoints = resolve_endpoints()
        registry = getattr(endpoints, "settlement_registry", None)
        rpc_url = getattr(endpoints, "rpc_url", None)
        escrow = getattr(endpoints, "escrow_pool", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("go-live: resolve_endpoints raised: %s", exc)

    if not registry:
        report.add("registry_resolved", "FAIL",
                   "no settlement registry resolved — set "
                   "PRSM_SETTLEMENT_REGISTRY_ADDRESS or PRSM_NETWORK.")
    elif registry.lower() == _RETIRED_REGISTRY.lower():
        report.add("registry_resolved", "FAIL",
                   f"resolved registry {registry} is the RETIRED pre-F registry "
                   "— re-point PRSM_SETTLEMENT_REGISTRY_ADDRESS at the F bundle.")
    else:
        report.add("registry_resolved", "PASS",
                   f"settlement registry {registry}")

    if escrow:
        report.add("escrow_resolved", "INFO", f"EscrowPool {escrow}")
    else:
        report.add("escrow_resolved", "WARN",
                   "no EscrowPool resolved — requester-funding pre-flight will "
                   "be skipped (finalizeBatch still enforces funding on-chain).")

    # ── on-chain reads (fail-soft via the reader) ───────────────────────────
    if reader is None and rpc_url:
        try:
            reader = _Web3Reader(rpc_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("go-live: reader build raised: %s", exc)
            reader = None

    if registry and reader is not None:
        paused = reader.paused(registry)
        if paused is True:
            report.add("registry_active", "FAIL",
                       f"registry {registry} is PAUSED — it rejects every "
                       "commitBatch. Point at an active (un-paused) F bundle.")
        elif paused is False:
            report.add("registry_active", "PASS", "registry is active (not paused)")
        else:
            report.add("registry_active", "WARN",
                       "could not read registry paused() (RPC error / not a "
                       "registry) — confirm the registry is live before go-live.")

        code = reader.code_hex(registry)
        try:
            from prsm.settlement.client_wiring import (
                _bytecode_exposes_selector,
                _commit_with_attestation_selector_hex,
            )
            sel = _commit_with_attestation_selector_hex()
        except Exception:  # noqa: BLE001
            sel = ""
        if code and sel and _bytecode_exposes_selector(code, sel):
            report.add("attestation_surface", "PASS",
                       "registry exposes commitBatchWithAttestation (roadmap F)")
        elif code:
            report.add("attestation_surface", "WARN",
                       "registry does NOT expose commitBatchWithAttestation — "
                       "settlement works but commits legacy commitBatch only "
                       "(no on-chain attestation; not the F bundle).")
        else:
            report.add("attestation_surface", "WARN",
                       "could not read registry bytecode to confirm the F surface.")
    elif registry:
        report.add("registry_active", "WARN",
                   "no RPC reader — skipped on-chain registry paused()/surface "
                   "checks; run with a reachable BASE_RPC_URL before go-live.")

    if signer and reader is not None:
        bal = reader.eth_balance_wei(signer)
        if bal is None:
            report.add("settler_funded", "WARN",
                       "could not read settler ETH balance (RPC error) — confirm "
                       "the settler key holds Base ETH for gas before go-live.")
        elif bal > 0:
            report.add("settler_funded", "PASS",
                       f"settler key holds {bal / 10 ** 18:.6f} ETH for gas")
        else:
            report.add("settler_funded", "WARN",
                       "settler key holds 0 ETH — fund it with Base ETH or "
                       "commitBatch/finalizeBatch cannot be broadcast.")

    # ── attestation flip readiness (offline + the surface read above) ───────
    flag_on = _truthy(environ.get("PRSM_SETTLEMENT_SUPPORTS_ATTESTATION"))
    surface_ok = any(
        f.check == "attestation_surface" and f.status == "PASS"
        for f in report.findings
    )
    if flag_on and surface_ok:
        report.add("attestation_flag", "PASS",
                   "PRSM_SETTLEMENT_SUPPORTS_ATTESTATION on + registry supports it "
                   "→ the first batch commits a real attestation (roadmap F live).")
    elif flag_on and not surface_ok:
        report.add("attestation_flag", "WARN",
                   "PRSM_SETTLEMENT_SUPPORTS_ATTESTATION on but the registry "
                   "surface is unconfirmed — the sp1299 fail-safe keeps "
                   "attestation OFF (legacy commitBatch) until confirmed.")
    else:
        report.add("attestation_flag", "INFO",
                   "PRSM_SETTLEMENT_SUPPORTS_ATTESTATION off — settlement commits "
                   "legacy commitBatch(bytes32(0)); set it to 1 to activate F.")

    return report


def run_multistage_go_live_preflight(
    *,
    env: Optional[dict] = None,
    reader: Optional[Any] = None,
    provider_address: Optional[str] = None,
    node: Optional[Any] = None,
) -> SettlementGoLiveReport:
    """Sprint 1337 (S4) — go-live preflight for BIG-MODEL MULTI-STAGE (Design A) on-chain
    settlement. Read-only/offline go/no-go, extending the single-stage
    ``run_settlement_go_live_preflight`` per-node.

    In Design A each stage node self-commits its OWN share with its OWN funded settler key, so
    the single-stage preflight (this node's settler key controls its provider_address + funded +
    registry active + write-capable client) IS the per-node gate. On top of it this adds the
    multi-stage config the per-stage delivery/split/commit path needs. Runnable standalone
    (``python -m prsm.settlement.go_live_preflight multistage``) using env only; pass a live
    ``node`` for the fuller per-stage client / receiver-store checks."""
    import os
    environ = env if env is not None else os.environ

    # Per-node base gate (settler key / funding / registry / write-capable client).
    report = run_settlement_go_live_preflight(
        env=environ, reader=reader, provider_address=provider_address)

    from prsm.settlement.client_wiring import multistage_settlement_enabled

    # ── the multi-stage gate ────────────────────────────────────────────────
    if multistage_settlement_enabled(environ):
        report.add("multistage_settlement", "PASS",
                   "PRSM_MULTISTAGE_SETTLEMENT enabled (per-stage delivery + self-commit active)")
    else:
        report.add("multistage_settlement", "FAIL",
                   "PRSM_MULTISTAGE_SETTLEMENT is off — the per-stage delivery hook returns [] "
                   "and no stage node self-commits; big-model paid settlement is inert.")

    # ── node_id → payee wallet map (the splitter fail-closes without it) ─────
    n = 0
    try:
        from prsm.node.compute_wallet_map import ComputeWalletMap
        n = len(getattr(ComputeWalletMap.from_env(environ), "_map", {}) or {})
    except Exception as exc:  # noqa: BLE001 — never crash the report
        logger.warning("multistage preflight: wallet map load raised: %s", exc)
    if n > 0:
        report.add("compute_wallet_map", "PASS",
                   f"PRSM_COMPUTE_WALLET_MAP_FILE resolves {n} node_id→payee entries")
    else:
        report.add("compute_wallet_map", "FAIL",
                   "no compute wallet map (PRSM_COMPUTE_WALLET_MAP_FILE empty/missing) — the "
                   "per-stage splitter fail-closes when a stage node_id has no payee address, so "
                   "the payee set can't be built and multi-stage settlement cannot commit.")

    # ── per-stage accumulate state (distinct from the single-stage client) ──
    ms_state = (environ.get("PRSM_MULTISTAGE_SETTLEMENT_STATE_FILE", "") or "").strip()
    report.add("per_stage_state_file", "INFO",
               (f"per-stage accumulate state at {ms_state} (count_threshold=1 client)"
                if ms_state else
                "PRSM_MULTISTAGE_SETTLEMENT_STATE_FILE unset — a DISTINCT default path is used "
                "(sp1329) so the per-stage client can't collide with the single-stage one."))

    # ── foreign-node task delivery endpoints ────────────────────────────────
    if (environ.get("PRSM_MULTISTAGE_ENDPOINT_MAP", "") or "").strip():
        report.add("endpoint_resolution", "INFO",
                   "PRSM_MULTISTAGE_ENDPOINT_MAP pins static stage-node endpoints (overrides "
                   "runtime discovery).")
    else:
        report.add("endpoint_resolution", "WARN",
                   "no static PRSM_MULTISTAGE_ENDPOINT_MAP — foreign per-node tasks rely on "
                   "runtime peer discovery (sp1309, provider-address match). Works live but is "
                   "unverifiable offline; confirm the stage peers are discoverable.")

    # ── prereq reminder: worker per-stage settlement signatures ─────────────
    report.add("worker_per_stage_sigs", "INFO",
               "worker per-stage settlement signatures are emitted at the sign sites (sp1158) "
               "when the settlement context is present — re-exercise the proven big-model runs "
               "with PRSM_MULTISTAGE_SETTLEMENT on to confirm they populate the receipt.")

    # ── richer per-stage wiring checks when a live node is supplied ─────────
    if node is not None:
        try:
            from prsm.settlement.client_wiring import (
                resolve_per_stage_receiver_store,
                resolve_per_stage_settlement_client,
            )
            ps_client = resolve_per_stage_settlement_client(node, environ)
            ps_write = bool(ps_client is not None and getattr(
                getattr(ps_client, "_contract", None), "address", None))
            report.add("per_stage_client", "PASS" if ps_write else "FAIL",
                       "per-stage settlement client builds WRITE-CAPABLE (count_threshold=1)"
                       if ps_write else
                       "per-stage settlement client did NOT build write-capable — check the "
                       "settler key / gate.")
            rx = resolve_per_stage_receiver_store(node, environ)
            report.add("per_stage_receiver_store", "PASS" if rx is not None else "WARN",
                       "per-stage receiver store wired (in-process ingest of this node's own task)"
                       if rx is not None else
                       "no per-stage receiver store — this node's OWN stage task falls back to a "
                       "self-HTTP POST instead of in-process ingest (sp1327).")
        except Exception as exc:  # noqa: BLE001
            report.add("per_stage_client", "WARN",
                       f"per-stage wiring check raised: {exc}")

    return report


def _main() -> int:
    import json
    import sys
    multistage = any(a in ("multistage", "--multistage") for a in sys.argv[1:])
    report = (run_multistage_go_live_preflight() if multistage
              else run_settlement_go_live_preflight())
    label = "MULTI-STAGE " if multistage else ""
    print(json.dumps(report.to_dict(), indent=2))
    for f in report.findings:
        print(f"  [{f.status:4}] {f.check}: {f.detail}")
    print(f"\n{label}{'✅ GO' if report.go else '❌ NO-GO'} "
          f"({sum(1 for f in report.findings if f.status == 'FAIL')} FAIL)")
    return 0 if report.go else 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
