"""Sprint 1081 — attestation collateral auto-refresh (Tier 3 ops completion).

Intel & AMD revocation lists (and the TCB-Info / QE-Identity collateral) carry a
``nextUpdate`` ~monthly. When a CRL goes stale, sp1060's ``_crl_current`` treats it as
"no CRL", so revocation is SILENTLY not enforced (a revoked TEE passes). This refreshes
the collateral from Intel PCS / AMD KDS on a schedule and caches it for the backends.

THE load-bearing safety property: a refresh must NEVER replace good collateral with
stale / invalid / garbage data. Every item is VALIDATED (signature + freshness)
BEFORE the atomic swap; on any bad fetch the existing cached copy is kept untouched.
So a transient network failure, a lagging/MITM'd response, or an expired upstream can
only ever leave the last-known-good collateral in place — never downgrade it.

The network fetch is injected (``fetch`` async callable) so the orchestration is
hermetically testable; ``httpx_fetch`` is the default real fetcher.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Optional, Tuple

from cryptography import x509

logger = logging.getLogger(__name__)

Fetch = Callable[[str], Awaitable[bytes]]
# Orchestration fetch returns (body, response-headers) so the issuer chain (carried in
# an Intel PCS response header) is available to validate the collateral against the root.
FetchWithHeaders = Callable[[str], Awaitable[Tuple[bytes, Mapping[str, str]]]]

# Intel Provisioning Certification Service (PCS) v4 — platform-independent collateral
# (no HWID / platform secret needed for the PCK CRLs or the global QE-Identity).
INTEL_PCS_BASE = "https://api.trustedservices.intel.com/sgx/certification/v4"
_INTEL_PCK_CRL_ISSUER_HEADER = "SGX-PCK-CRL-Issuer-Chain"
_INTEL_QE_ISSUER_HEADER = "SGX-Enclave-Identity-Issuer-Chain"
_INTEL_TCB_INFO_ISSUER_HEADER = "TCB-Info-Issuer-Chain"

# Cached-collateral file names (shared by the refresher [writer] and the DCAP backend
# build [reader] so they agree on the layout under the collateral cache dir).
INTEL_PCK_PLATFORM_CRL_FILE = "intel_pck_platform.crl.pem"
INTEL_PCK_PROCESSOR_CRL_FILE = "intel_pck_processor.crl.pem"
INTEL_QE_IDENTITY_FILE = "intel_qe_identity.json"


def intel_tcb_info_cache_file(fmspc: str) -> str:
    return f"intel_tcb_{fmspc.lower()}.json"


def _is_valid_fmspc(fmspc: str) -> bool:
    """An Intel FMSPC is exactly 6 bytes = 12 hex chars. Validating it before it reaches
    a URL / cache filename removes any path/query metacharacter surface (review #4)."""
    return (isinstance(fmspc, str) and len(fmspc) == 12
            and all(c in "0123456789abcdef" for c in fmspc))


def _configured_fmspc(environ=None) -> Optional[str]:
    environ = environ if environ is not None else os.environ
    raw = (environ.get("PRSM_INTEL_SGX_FMSPC", "") or "").strip().lower()
    return raw if _is_valid_fmspc(raw) else None

# AMD KDS (kdsintf.amd.com) — public per-product VCEK CRL + cert_chain (no platform
# secret needed). KDS uses Capitalized product names in the URL path.
AMD_KDS_BASE = "https://kdsintf.amd.com"
_AMD_PRODUCTS = ("milan", "genoa", "turin")


def amd_crl_cache_file(product: str) -> str:
    return f"amd_{product.lower()}.crl.pem"


def collateral_cache_dir(environ=None) -> Optional[Path]:
    """The attestation-collateral cache dir, or None when unset. Set by the node from
    its data_dir; ``PRSM_ATTESTATION_COLLATERAL_DIR`` overrides."""
    environ = environ if environ is not None else os.environ
    raw = (environ.get("PRSM_ATTESTATION_COLLATERAL_DIR", "") or "").strip()
    return Path(raw) if raw else None


def read_cached_intel_crls(environ=None) -> Optional[bytes]:
    """Concatenated PEM of the cached Intel PCK CRLs (platform + processor) the refresher
    wrote, or None if the cache dir is unset / no CRL has been refreshed yet."""
    cache = collateral_cache_dir(environ)
    if cache is None:
        return None
    expected = (INTEL_PCK_PLATFORM_CRL_FILE, INTEL_PCK_PROCESSOR_CRL_FILE)
    parts = []
    missing = []
    for name in expected:
        try:
            parts.append((cache / name).read_bytes())
        except OSError:
            missing.append(name)
    if missing and parts:
        # sp1081 review H1 — a partial CRL set means revocation is enforced for ONLY
        # the CA(s) whose CRL is present; the missing one silently isn't checked
        # (require_crl=False). Surface it loudly rather than presenting a partial set
        # as if it were complete.
        logger.warning("Intel PCK CRL cache is INCOMPLETE (%d/%d present; missing %s) — "
                       "revocation is NOT enforced for the missing CA until it refreshes",
                       len(parts), len(expected), ", ".join(missing))
    return b"\n".join(parts) if parts else None


def read_cached_amd_crls(environ=None) -> Optional[bytes]:
    """Concatenated PEM of the cached AMD VCEK CRLs (one per product the refresher has
    fetched), or None if the cache dir is unset / no AMD CRL has been refreshed yet."""
    cache = collateral_cache_dir(environ)
    if cache is None:
        return None
    parts = []
    present = []
    for product in _AMD_PRODUCTS:
        try:
            parts.append((cache / amd_crl_cache_file(product)).read_bytes())
            present.append(product)
        except OSError:
            continue
    if parts:
        # sp1082 review L4 — name which products have a live CRL so an operator can tell
        # which platforms have revocation enforced (a missing product only affects that
        # product's nodes — AMD products are independent platforms, unlike Intel's H1).
        logger.info("AMD VCEK CRL cache present for: %s (revocation enforced for these)",
                    ", ".join(present))
    return b"\n".join(parts) if parts else None


def read_cached_intel_qe_identity(environ=None) -> Optional[bytes]:
    """The cached Intel QE-Identity JSON, or None. Usable by the backend only when the
    TCB Signing cert is also configured (the QE signer)."""
    cache = collateral_cache_dir(environ)
    if cache is None:
        return None
    try:
        return (cache / INTEL_QE_IDENTITY_FILE).read_bytes()
    except OSError:
        return None


def read_cached_intel_tcb_info(environ=None) -> Optional[bytes]:
    """The cached per-FMSPC Intel TCB-Info JSON for the configured PRSM_INTEL_SGX_FMSPC,
    or None (no fmspc configured / no cache dir / not refreshed yet). Usable by the
    backend only when the TCB Signing cert is also configured (its signer)."""
    cache = collateral_cache_dir(environ)
    fmspc = _configured_fmspc(environ)
    if cache is None or not fmspc:
        return None
    try:
        return (cache / intel_tcb_info_cache_file(fmspc)).read_bytes()
    except OSError:
        return None


@dataclass(frozen=True)
class RefreshResult:
    ok: bool
    cached: bool = False
    reason: Optional[str] = None


def _load_crl_any(data: bytes):
    """Parse a CRL from PEM or DER (AMD KDS serves DER, Intel PCS PEM). None on failure."""
    try:
        return x509.load_pem_x509_crl(data)
    except Exception:  # noqa: BLE001
        pass
    try:
        return x509.load_der_x509_crl(data)
    except Exception:  # noqa: BLE001
        return None


def _crl_to_pem(data: bytes) -> Optional[bytes]:
    """Normalize a CRL (PEM or DER) to PEM bytes so the backends' PEM-split CRL loaders
    can read it uniformly. None if it doesn't parse as a CRL."""
    crl = _load_crl_any(data)
    if crl is None:
        return None
    from cryptography.hazmat.primitives import serialization
    return crl.public_bytes(serialization.Encoding.PEM)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomic write: tmp + fsync(file) + rename + fsync(dir) (mirrors the sp1039 store)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)   # atomic on POSIX
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):  # pragma: no cover - platform-specific
        pass


def crl_is_valid_and_current(pem: bytes, issuer_cert: x509.Certificate,
                             now: Optional[_dt.datetime] = None) -> bool:
    """True iff ``pem`` parses as a CRL that is signature-valid under ``issuer_cert``
    AND currently within [thisUpdate, nextUpdate]. Total — never raises."""
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    crl = _load_crl_any(pem)
    if crl is None:
        return False
    try:
        if not crl.is_signature_valid(issuer_cert.public_key()):
            return False
    except Exception:  # noqa: BLE001
        return False
    try:
        lu = crl.last_update_utc
        nu = crl.next_update_utc
    except AttributeError:  # pragma: no cover - cryptography < 42
        lu = crl.last_update.replace(tzinfo=_dt.timezone.utc)
        nu = (crl.next_update.replace(tzinfo=_dt.timezone.utc)
              if crl.next_update is not None else None)
    if lu is not None and now < lu:
        return False
    if nu is not None and now > nu:
        return False
    return True


def tcb_info_is_valid(json_bytes: bytes, signer_cert: x509.Certificate) -> bool:
    """True iff the signed Intel TCB-Info verifies against ``signer_cert`` (reuses the
    sp1062 verifier). Total — never raises. (Signature only; see
    ``tcb_info_is_valid_and_current`` for the freshness-gated form.)"""
    try:
        from prsm.compute.inference.sgx_tcb import parse_and_verify_tcb_info
        parse_and_verify_tcb_info(json_bytes, signer_cert)
        return True
    except Exception:  # noqa: BLE001
        return False


def _tcb_info_is_current(json_bytes: bytes, now: _dt.datetime) -> bool:
    """sp1089 — freshness gate for a signed TCB-Info: parse issueDate / nextUpdate from
    the VERIFIED tcbInfo bytes (not the raw blob — a raw decoy could carry an unsigned
    duplicate) and require now within [issueDate, nextUpdate]. A correctly-signed-but-
    stale TCB-Info must be rejected so the sp1061-1063 recency check can't keep passing a
    TEE Intel has since marked OutOfDate. Absent nextUpdate → fail (we promise freshness)."""
    import json as _json
    from prsm.compute.inference.sgx_tcb import _extract_tcb_info_bytes
    try:
        body = _json.loads(_extract_tcb_info_bytes(json_bytes))
    except Exception:  # noqa: BLE001
        return False

    def _parse(ts):
        if not isinstance(ts, str):
            return None
        try:
            return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    issue = _parse(body.get("issueDate"))
    nxt = _parse(body.get("nextUpdate"))
    if nxt is None:
        return False
    if issue is not None and now < issue:
        return False
    return now <= nxt


def tcb_info_is_valid_and_current(json_bytes: bytes, signer_cert: x509.Certificate,
                                  now: Optional[_dt.datetime] = None) -> bool:
    """True iff the signed Intel TCB-Info verifies against ``signer_cert`` AND is within
    its [issueDate, nextUpdate] freshness window (sp1089)."""
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    if not tcb_info_is_valid(json_bytes, signer_cert):
        return False
    return _tcb_info_is_current(json_bytes, now)


def _qe_identity_is_current(json_bytes: bytes, now: _dt.datetime) -> bool:
    """sp1081 review M2 — freshness gate for a signed QE-Identity: parse issueDate /
    nextUpdate from the VERIFIED enclaveIdentity bytes (not the raw blob — the raw blob
    could carry an unsigned decoy) and require now within [issueDate, nextUpdate]. A
    correctly-signed-but-replayed-stale QE-Identity must be rejected so it can't
    under-state the ISVSVN floor. Absent fields → fail (the sprint promises freshness)."""
    import json as _json
    from prsm.compute.inference.qe_identity import _extract_enclave_identity_bytes
    try:
        body = _json.loads(_extract_enclave_identity_bytes(json_bytes))
    except Exception:  # noqa: BLE001
        return False

    def _parse(ts):
        if not isinstance(ts, str):
            return None
        try:
            return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    issue = _parse(body.get("issueDate"))
    nxt = _parse(body.get("nextUpdate"))
    if nxt is None:
        return False   # no freshness horizon → can't bound staleness → refuse
    if issue is not None and now < issue:
        return False
    return now <= nxt


def qe_identity_is_valid(json_bytes: bytes, signer_cert: x509.Certificate,
                         now: Optional[_dt.datetime] = None) -> bool:
    """True iff the signed Intel QE Identity verifies against ``signer_cert`` (sp1079)
    AND is currently within its [issueDate, nextUpdate] freshness window."""
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    try:
        from prsm.compute.inference.qe_identity import parse_and_verify_qe_identity
        parse_and_verify_qe_identity(json_bytes, signer_cert)
    except Exception:  # noqa: BLE001
        return False
    return _qe_identity_is_current(json_bytes, now)


def _cache_if_valid(data: bytes, validate: Callable[[bytes], bool],
                    cache_path: Path, *, label: str) -> RefreshResult:
    """The validate-before-swap core: write ``data`` to ``cache_path`` ONLY if it
    validates; otherwise keep the existing cached copy untouched. Never raises."""
    if not isinstance(data, (bytes, bytearray)) or not data:
        return RefreshResult(ok=False, cached=False, reason="empty-response")
    try:
        valid = bool(validate(bytes(data)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("collateral validation raised for %s: %s", label, exc)
        valid = False
    if not valid:
        logger.warning("refreshed collateral (%s) is invalid/stale — keeping the existing "
                       "cached copy (revocation/recency NOT downgraded)", label)
        return RefreshResult(ok=False, cached=False, reason="invalid-or-stale")
    try:
        _atomic_write_bytes(Path(cache_path), bytes(data))
    except OSError as exc:
        logger.error("could not cache validated collateral to %s: %s", cache_path, exc)
        return RefreshResult(ok=False, cached=False, reason=f"cache-write-failed: {exc}")
    logger.info("refreshed attestation collateral (%s) → %s (%d bytes)",
                label, cache_path, len(data))
    return RefreshResult(ok=True, cached=True)


async def refresh_collateral_item(
    *,
    fetch: Fetch,
    url: str,
    validate: Callable[[bytes], bool],
    cache_path: Path,
) -> RefreshResult:
    """Fetch one collateral item, VALIDATE it, and only on success atomically write it
    to ``cache_path``. On a fetch error or a failed validation the existing cached copy
    (if any) is left untouched. Never raises."""
    try:
        data = await fetch(url)
    except Exception as exc:  # noqa: BLE001 - a fetch failure must never crash the loop
        logger.warning("collateral fetch failed for %s: %s — keeping cached copy", url, exc)
        return RefreshResult(ok=False, cached=False, reason=f"fetch-failed: {exc}")
    return _cache_if_valid(data, validate, Path(cache_path), label=url)


# ── issuer-chain helpers ─────────────────────────────────────────────────────────

def _split_pem_certs(pem_bytes: bytes) -> list:
    """Parse a concatenated PEM blob into a list of certificates (in file order)."""
    marker = b"-----BEGIN CERTIFICATE-----"
    out = []
    for part in pem_bytes.split(marker)[1:]:
        try:
            out.append(x509.load_pem_x509_certificate(marker + part))
        except Exception:  # noqa: BLE001
            continue
    return out


def _chain_from_issuer_header(headers: Mapping[str, str], header_name: str) -> list:
    """Extract + URL-decode the issuer cert chain Intel PCS carries in a response
    header ([signing leaf, ...intermediates]). Case-insensitive lookup; [] if absent."""
    value = None
    for k, v in (headers or {}).items():
        if k.lower() == header_name.lower():
            value = v
            break
    if not value:
        return []
    return _split_pem_certs(urllib.parse.unquote(value).encode())


async def httpx_fetch(url: str, *, timeout: float = 30.0) -> bytes:
    """Default body-only fetcher (lazy httpx import). Raises on non-2xx / transport
    error so refresh_collateral_item keeps the cached copy."""
    body, _headers = await httpx_fetch_with_headers(url, timeout=timeout)
    return body


async def httpx_fetch_with_headers(url: str, *, timeout: float = 30.0):
    """Default fetcher returning (body, headers). Raises on non-2xx / transport error."""
    import httpx
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content, dict(resp.headers)


# ── orchestration: Intel SGX collateral refresh ──────────────────────────────────

class CollateralRefresher:
    """Refreshes the platform-independent Intel SGX attestation collateral (the PCK
    revocation lists + the global QE-Identity) from Intel PCS into a cache dir that the
    DCAP backend reads. Every item is validated — issuer chain to the (bundled) Intel
    SGX Root CA, then the collateral's own signature + freshness — BEFORE the atomic
    swap, so a refresh can never downgrade good cached collateral.

    AMD KDS CRL refresh is a documented follow-on: the AMD CRL is ASK-signed (an
    intermediate, RSA-PSS, DER), so it needs a KDS cert_chain fetch + the bundled ARK,
    a heavier path than the Intel issuer-chain-in-header model used here.

    TCB-Info (sp1089) IS refreshed per-FMSPC when the operator configures the platform
    FMSPC (``intel_fmspc`` / PRSM_INTEL_SGX_FMSPC); absent → skipped (it's platform-specific).
    """

    def __init__(self, cache_dir, *, fetch: Optional[FetchWithHeaders] = None,
                 intel_root_pem: Optional[bytes] = None, now_fn=None,
                 intel_pcs_base: str = INTEL_PCS_BASE,
                 tcb_signer_pem: Optional[bytes] = None,
                 amd_ark_pems: Optional[dict] = None,
                 amd_kds_base: str = AMD_KDS_BASE,
                 intel_fmspc: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        # sp1089 — the platform FMSPC (12-hex) whose TCB-Info to refresh; None disables
        # TCB-Info refresh (it's per-platform, so the operator supplies it).
        self._intel_fmspc = (intel_fmspc or "").strip().lower() or None
        self._fetch = fetch or httpx_fetch_with_headers
        self._intel_root_pem = intel_root_pem
        self._now_fn = now_fn or (lambda: _dt.datetime.now(_dt.timezone.utc))
        self._intel_pcs_base = intel_pcs_base.rstrip("/")
        # sp1082 — bundled (fingerprint-pinned) AMD ARKs per product; the trust anchor
        # for the KDS-supplied ASK that signs the VCEK CRL. Default → bundled_amd_arks().
        self._amd_ark_pems = amd_ark_pems
        self._amd_kds_base = amd_kds_base.rstrip("/")
        # sp1081 review M1 — the operator's configured QE signer (the same TCB Signing
        # cert the verify path uses). When set, the refreshed QE-Identity is validated
        # under IT (not just the header-chain leaf) so a refresh can't cache a QE the
        # verify path would then reject on every attestation (availability poison).
        self._tcb_signer_pem = tcb_signer_pem

    def _intel_root(self) -> x509.Certificate:
        pem = self._intel_root_pem
        if not pem:
            from prsm.compute.inference.vendor_anchors import bundled_intel_sgx_root_ca
            pem = bundled_intel_sgx_root_ca()
        return x509.load_pem_x509_certificate(pem)

    async def _refresh_chain_validated(self, *, url: str, header_name: str,
                                       cache_name: str,
                                       validate_under_leaf) -> RefreshResult:
        """Fetch (body, headers); validate the issuer chain (header) to the Intel root,
        then validate the body under the chain's leaf; cache only if both pass."""
        try:
            body, headers = await self._fetch(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("collateral fetch failed for %s: %s — keeping cached copy",
                           url, exc)
            return RefreshResult(ok=False, cached=False, reason=f"fetch-failed: {exc}")

        from prsm.compute.inference.x509_path import verify_cert_chain

        def validate(data: bytes) -> bool:
            chain = _chain_from_issuer_header(headers, header_name)
            if not chain:
                logger.warning("no %s issuer chain on %s — refusing the refresh",
                               header_name, url)
                return False
            try:
                root = self._intel_root()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intel root unusable for collateral refresh: %s", exc)
                return False
            ok, err = verify_cert_chain(chain, root, now=self._now_fn(),
                                        require_crl=False)
            if not ok:
                logger.warning("issuer chain for %s does not chain to the Intel root: %s",
                               url, err)
                return False
            return bool(validate_under_leaf(data, chain[0]))

        return _cache_if_valid(body, validate, self.cache_dir / cache_name, label=url)

    async def refresh_intel_pck_crl(self, ca: str) -> RefreshResult:
        """Refresh the Intel PCK CRL for ``ca`` ('platform' or 'processor')."""
        url = f"{self._intel_pcs_base}/pckcrl?ca={ca}&encoding=pem"
        return await self._refresh_chain_validated(
            url=url, header_name=_INTEL_PCK_CRL_ISSUER_HEADER,
            cache_name=f"intel_pck_{ca}.crl.pem",
            validate_under_leaf=lambda data, leaf: crl_is_valid_and_current(
                data, leaf, self._now_fn()))

    async def refresh_intel_qe_identity(self) -> RefreshResult:
        """Refresh the global Intel QE-Identity (enclaveIdentity)."""
        url = f"{self._intel_pcs_base}/qe/identity"

        def _validate(data, leaf):
            # Validate under the operator-pinned TCB signer when configured (matches the
            # verify path); else under the header-chain leaf (already chained to the
            # pinned root by _refresh_chain_validated).
            signer = leaf
            if self._tcb_signer_pem:
                try:
                    signer = x509.load_pem_x509_certificate(self._tcb_signer_pem)
                except Exception:  # noqa: BLE001
                    return False
            return qe_identity_is_valid(data, signer, self._now_fn())

        return await self._refresh_chain_validated(
            url=url, header_name=_INTEL_QE_ISSUER_HEADER,
            cache_name="intel_qe_identity.json", validate_under_leaf=_validate)

    def _tcb_signer_or_leaf(self, leaf):
        """The operator-pinned TCB Signing cert when configured (matches the verify
        path), else the header-chain leaf. None signals a bad configured signer."""
        if not self._tcb_signer_pem:
            return leaf
        try:
            return x509.load_pem_x509_certificate(self._tcb_signer_pem)
        except Exception:  # noqa: BLE001
            return None

    async def refresh_intel_tcb_info(self, fmspc: str) -> RefreshResult:
        """sp1089 — refresh the per-FMSPC Intel TCB-Info. Validated like the QE-Identity:
        issuer chain (TCB-Info-Issuer-Chain header) to the bundled Intel root, then the
        signed TCB-Info under the configured TCB signer (or the header leaf) + its
        [issueDate, nextUpdate] freshness window — so a stale TCB-Info can't keep the
        recency check (sp1061-1063) passing a now-OutOfDate TEE."""
        fmspc = (fmspc or "").strip().lower()
        if not _is_valid_fmspc(fmspc):
            return RefreshResult(ok=False, cached=False, reason="invalid-fmspc")
        url = f"{self._intel_pcs_base}/tcb?fmspc={fmspc}"

        def _validate(data, leaf):
            signer = self._tcb_signer_or_leaf(leaf)
            if signer is None:
                return False
            return tcb_info_is_valid_and_current(data, signer, self._now_fn())

        return await self._refresh_chain_validated(
            url=url, header_name=_INTEL_TCB_INFO_ISSUER_HEADER,
            cache_name=intel_tcb_info_cache_file(fmspc), validate_under_leaf=_validate)

    def _amd_arks(self) -> dict:
        if self._amd_ark_pems is not None:
            return self._amd_ark_pems
        from prsm.compute.inference.vendor_anchors import bundled_amd_arks
        return bundled_amd_arks()

    async def refresh_amd_crl(self, product: str) -> RefreshResult:
        """Refresh the AMD VCEK CRL for ``product`` ('milan'/'genoa'/'turin'). The CRL
        (DER from KDS) is signed by the ASK; the ASK is fetched from the KDS cert_chain
        endpoint and validated to the BUNDLED, fingerprint-pinned ARK (never a
        KDS-supplied root), then the CRL is validated under the ASK + cached as PEM."""
        product = product.lower()
        arks = self._amd_arks()
        ark_pem = arks.get(product)
        if not ark_pem:
            return RefreshResult(ok=False, cached=False, reason=f"no-ark-for-{product}")
        kds_product = product.capitalize()
        crl_url = f"{self._amd_kds_base}/vcek/v1/{kds_product}/crl"
        chain_url = f"{self._amd_kds_base}/vcek/v1/{kds_product}/cert_chain"
        try:
            crl_body, _ = await self._fetch(crl_url)
            chain_body, _ = await self._fetch(chain_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AMD KDS fetch failed for %s: %s — keeping cached copy",
                           product, exc)
            return RefreshResult(ok=False, cached=False, reason=f"fetch-failed: {exc}")

        # KDS serves the CRL as DER — normalize to PEM so the backend's PEM-split loader
        # reads it, and so validate-before-swap operates on the cached bytes.
        crl_pem = _crl_to_pem(crl_body)
        if crl_pem is None:
            return RefreshResult(ok=False, cached=False, reason="unparseable-crl")

        from prsm.compute.inference.x509_path import verify_cert_chain

        def validate(data: bytes) -> bool:
            chain = _split_pem_certs(chain_body)   # [ASK, ARK] from KDS
            if not chain:
                logger.warning("no ASK chain from KDS for %s — refusing refresh", product)
                return False
            try:
                ark = x509.load_pem_x509_certificate(ark_pem)
            except Exception as exc:  # noqa: BLE001
                logger.warning("bundled AMD ARK unusable for %s: %s", product, exc)
                return False
            # Validate ONLY the ASK (chain[0]) to OUR pinned ARK — discard the
            # KDS-supplied ARK; we never trust a self-supplied root.
            ok, err = verify_cert_chain([chain[0]], ark, now=self._now_fn(),
                                        require_crl=False)
            if not ok:
                logger.warning("AMD ASK for %s does not chain to the bundled ARK: %s",
                               product, err)
                return False
            return crl_is_valid_and_current(data, chain[0], self._now_fn())

        return _cache_if_valid(crl_pem, validate,
                               self.cache_dir / amd_crl_cache_file(product),
                               label=crl_url)

    async def refresh_all(self) -> dict:
        """Refresh every supported item; return {name: RefreshResult}. Never raises."""
        results: dict = {}
        for ca in ("platform", "processor"):
            results[f"intel_pck_{ca}_crl"] = await self.refresh_intel_pck_crl(ca)
        results["intel_qe_identity"] = await self.refresh_intel_qe_identity()
        # sp1089 — per-FMSPC TCB-Info, only when the operator configured the platform fmspc.
        if self._intel_fmspc:
            results[f"intel_tcb_info_{self._intel_fmspc}"] = \
                await self.refresh_intel_tcb_info(self._intel_fmspc)
        # AMD VCEK CRLs — only products whose ARK is available (skip the rest quietly).
        try:
            available = self._amd_arks()
        except Exception as exc:  # noqa: BLE001
            available = {}
            logger.debug("AMD ARKs unavailable for refresh: %s", exc)
        for product in _AMD_PRODUCTS:
            if available.get(product):
                results[f"amd_{product}_crl"] = await self.refresh_amd_crl(product)
        ok = sum(1 for r in results.values() if r.ok)
        logger.info("collateral refresh complete: %d/%d items updated", ok, len(results))
        return results
