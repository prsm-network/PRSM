# TEE Attestation — PKI Roots & Collateral Operator Runbook

**Date:** 2026-06-24 · **Audience:** node operators enabling hardware-verified TEE
attestation (Intel SGX DCAP / AMD SEV-SNP) · **Hardware required:** none for this
setup (PKI + collateral are software config; the hardware quote *generation* on a
serving node is a separate concern — see the TEE Tier-3 roadmap).

This runbook covers the one-time, no-hardware setup an operator does to make a
PRSM node **verify** TEE attestations against real vendor PKI, keep the revocation
/recency **collateral** fresh, and **observe** that enforcement is actually working.

---

## 1. Why this matters (trust model)

`verify_attestation` routes each quote through the `AttestationBackendRegistry`.
Two outcomes:

- **Structural backends** (`IntelASPBackend`/`AMDKDSBackend`/`DevOnlyBackend`) parse
  the quote but return `vendor_verified=False` — they prove *shape*, not *origin*.
- **Real verified backends** (`IntelDCAPBackend`/`AMDSEVSNPBackend`) chain the quote
  to a configured vendor **root** and return `vendor_verified=True`.

`vendor_verified` is **only as strong as the root you anchor to.** A real verifier
also checks, when the collateral is present:

- **Revocation** — the cached CRLs (Intel PCK platform/processor + root-CA CRL; AMD
  per-product VCEK CRL). A revoked intermediate/leaf is rejected.
- **Recency** — the signed Intel TCB-Info + QE-Identity, each gated to *now* within
  its `[issueDate, nextUpdate]` window, so a down-level (out-of-date) TEE is rejected.

**The failure mode this runbook prevents:** the collateral silently goes stale (the
auto-refresh stops succeeding), its `nextUpdate` lapses, and the freshness gates
start rejecting good quotes — or, worse, revocation stops being enforced — with no
operator visibility. §4 wires the metric + alert that closes that gap.

---

## 2. One-time PKI setup (roots)

PRSM **ships the genuine roots bundled and fingerprint-pinned** (Intel SGX Root CA;
AMD ARKs for Milan/Genoa/Turin) — see `prsm/compute/inference/vendor_anchors.py`.
They are **on by default**, so for the common case you install *nothing*: enabling a
verifier just requires the bundled root to be active (it is) plus the per-vendor
recency material below.

| Purpose | Env var | Default / notes |
|---|---|---|
| Intel SGX Root CA (custom) | `PRSM_INTEL_SGX_ROOT_CA_PEM` / `_FILE` | optional — overrides the bundled root |
| Use bundled Intel root | `PRSM_INTEL_SGX_USE_BUNDLED_ROOT` | `1` (default on); set `0` to require a custom root |
| AMD ARK (custom) | `PRSM_AMD_SEV_SNP_ARK_PEM` / `_FILE` | optional — overrides the bundled ARK |
| Use bundled AMD ARK | `PRSM_AMD_SEV_SNP_USE_BUNDLED_ROOT` | `1` (default on); set `0` to require a custom root |

> Supply a custom root only if you operate against a non-standard PKI (e.g. a test
> harness). For production Intel/AMD hardware, the bundled roots are correct — do
> **not** disable them and leave nothing configured (that downgrades to structural,
> `vendor_verified=False`, silently).

### Intel recency material (enables the TCB/QE-Identity gates)

| Purpose | Env var |
|---|---|
| Platform FMSPC (selects the per-platform TCB-Info) | `PRSM_INTEL_SGX_FMSPC` (12 hex chars) |
| TCB / QE-Identity signer cert | `PRSM_INTEL_SGX_TCB_SIGNING_PEM` / `_FILE` |
| Seed CRL / TCB-Info / QE-Identity (optional; auto-refresh keeps them current) | `PRSM_INTEL_SGX_CRL_PEM`/`_FILE`, `PRSM_INTEL_SGX_TCB_INFO_JSON`/`_FILE`, `PRSM_INTEL_SGX_QE_IDENTITY_JSON`/`_FILE` |

### AMD policy material (optional hardening)

| Purpose | Env var |
|---|---|
| Minimum reported TCB | `PRSM_AMD_SEV_SNP_MIN_TCB` |
| VMPL policy | `PRSM_AMD_SEV_SNP_VMPL_POLICY` |
| Seed VCEK CRL (optional) | `PRSM_AMD_SEV_SNP_CRL_PEM` / `_FILE` |

`configure_default_registry_from_env()` is called once at node startup and is
**fail-open**: with no anchor configured it leaves the structural behavior unchanged
(no crash), and returns `True` iff a real verifier was activated.

---

## 3. Collateral cache + auto-refresh (sp1081/1082)

CRLs (~monthly `nextUpdate`) and TCB-Info must be refreshed or they go stale. The
node refreshes them from **Intel PCS** (`https://api.trustedservices.intel.com/sgx/certification/v4`)
and **AMD KDS** (`https://kdsintf.amd.com`) on a schedule and caches them where the
verifiers read.

| Purpose | Env var | Default |
|---|---|---|
| Collateral cache dir (**enables** the whole feature) | `PRSM_ATTESTATION_COLLATERAL_DIR` | unset → loop idle |
| Mark auto-refresh enabled (status/observability flag) | `PRSM_COLLATERAL_AUTO_REFRESH` | off |
| Refresh interval (seconds) | `PRSM_COLLATERAL_REFRESH_INTERVAL_S` | `86400` (daily), floored at 60 |

What refreshes each cycle: Intel PCK platform + processor CRLs, the Intel root-CA
CRL, the QE-Identity, and the per-FMSPC TCB-Info; AMD per-product VCEK CRLs.

**Safety property (load-bearing):** every fetched item is **validated** (issuer chain
to the configured root + signature + `[thisUpdate, nextUpdate]` freshness) **before**
an atomic swap into the cache. A bad/stale/garbage fetch is rejected and the existing
good cached copy is kept untouched — a refresh can **never** downgrade good collateral.
After a successful refresh the affected backend is rebuilt in place (one atomic list
rebind, no transient zero-backend window), so a long-running node picks up fresh
revocation data **without a restart**.

**Egress requirement:** the node needs outbound HTTPS to the Intel PCS and AMD KDS
hosts above. If your firewall blocks them, the refresh fails and the collateral goes
stale — which §4 makes visible.

---

## 4. Verifying enforcement is live (and staying live)

### Human view — `/health/detailed`

`collateral_refresh_status` reports, per cached item: `present`, `fresh` (now within
its real `nextUpdate`), `age_seconds`, and the resolved `next_update`. Use it for a
point-in-time check after setup.

### Machine view — metrics + alert (sp1244)

The health JSON alone can't page you. Enable the node-runtime observability stack so
the collateral freshness becomes a Prometheus signal the AlertManager watches:

| Purpose | Env var |
|---|---|
| Expose `/metrics` (NodeRuntimeMetrics) | `PRSM_RUNTIME_METRICS_ENABLED=1` |
| Register the node-runtime alert rules | `PRSM_RUNTIME_ALERTS_ENABLED=1` |
| Deliver fired alerts to a webhook (optional) | `PRSM_ALERT_WEBHOOK_URL=<url>` |

Emitted **only when `PRSM_ATTESTATION_COLLATERAL_DIR` is set** (non-TEE nodes stay
silent):

- `prsm_collateral_refresh_enabled` — `1` when `PRSM_COLLATERAL_AUTO_REFRESH` is on.
- `prsm_collateral_age_seconds{item="…"}` — age (file mtime) of each cached item.
- `prsm_collateral_item_stale{item="…"}` — per-item: `1` when that item is **past its
  `nextUpdate` or has an unparseable horizon** (enforcement can't be trusted), else
  `0`. Horizon-aware — uses each item's real `nextUpdate`, not a fixed age threshold.
- `prsm_collateral_stale` — **unlabeled** max-over-items aggregate (`1` if *any* item
  is stale). This is the **alert target**: the AlertManager reduces a metric to one
  series, so the per-item gauge can't express "any item stale" — this scalar can.

**Alert rule `attestation_collateral_stale`** fires `WARNING` when
`prsm_collateral_stale` is `> 0` sustained for **1 hour** (`aggregation=max`). The
1-hour window debounces a transient collect/parse hiccup; a real staleness persists
because the refresh loop is daily and CRLs are monthly — so the alert gives ample
runway to act before the freshness gates begin rejecting traffic.

---

## 5. Remediation — `attestation_collateral_stale` is firing

Work top-down; the cause is almost always one of:

1. **Egress blocked** — the node can't reach Intel PCS / AMD KDS. Check outbound HTTPS
   to the §3 hosts. This is the most common cause on a freshly firewalled host.
2. **Cache dir not writable** — `PRSM_ATTESTATION_COLLATERAL_DIR` must be writable by
   the node user (the refresher does a tmp-write + atomic rename + dir fsync).
3. **Misconfigured recency inputs** — a stale **TCB-Info** specifically needs a valid
   `PRSM_INTEL_SGX_FMSPC` (12 hex chars) *and* `PRSM_INTEL_SGX_TCB_SIGNING_*` so the
   refreshed TCB-Info validates under the same signer the verify path uses; a stale
   **QE-Identity** likewise needs the TCB signer.
4. **Refresh loop idle** — `PRSM_ATTESTATION_COLLATERAL_DIR` unset means the loop
   never runs (you'd see the `loop idle` log line and no collateral metrics at all).

Confirm recovery on `/health/detailed` (item flips back to `fresh: true`) — the alert
clears once `prsm_collateral_stale` returns to `0` for the rule's window.

---

## 6. Quick reference — minimal production enablement

```bash
# Verify Intel SGX DCAP (bundled root on by default) with full recency + revocation:
export PRSM_INTEL_SGX_FMSPC=<12-hex-fmspc>
export PRSM_INTEL_SGX_TCB_SIGNING_FILE=/etc/prsm/intel_tcb_signer.pem
# Verify AMD SEV-SNP (bundled ARK on by default):
export PRSM_AMD_SEV_SNP_MIN_TCB=<min-tcb>          # optional hardening
# Keep collateral fresh:
export PRSM_ATTESTATION_COLLATERAL_DIR=/var/lib/prsm/attestation-collateral
export PRSM_COLLATERAL_AUTO_REFRESH=1
# Observe it (metrics + paging):
export PRSM_RUNTIME_METRICS_ENABLED=1
export PRSM_RUNTIME_ALERTS_ENABLED=1
export PRSM_ALERT_WEBHOOK_URL=https://hooks.example.com/prsm-alerts
```

Roots ship bundled and fingerprint-pinned, so the above is the whole one-time setup;
no PKI material is hand-installed unless you deliberately override a vendor root.
