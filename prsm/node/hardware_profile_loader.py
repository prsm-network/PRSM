"""Sprint 681 — load the local hardware_profile for DISCOVERY_ANNOUNCE.

Resolution order:
  1. ``PRSM_HARDWARE_PROFILE_FILE`` env → operator-pinned JSON path.
     If set but file missing/invalid → return None (operator was
     being explicit; don't silently fall through).
  2. ``<cache_dir>/hardware_profile.json`` — cache from prior runs.
  3. Fresh compute via the supplied (or default) profiler factory.
     Result is cached to <cache_dir>/hardware_profile.json for
     next start (write failure is non-fatal).

Any unexpected error returns None. Pre-680 wire format is preserved
when None is propagated through sprint 680's announce_self() path.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".prsm"
_CACHE_FILENAME = "hardware_profile.json"


def _default_profiler_factory():
    from prsm.compute.wasm.profiler import HardwareProfiler
    return HardwareProfiler()


def load_local_hardware_profile(
    cache_dir: Optional[Path] = None,
    profiler_factory: Callable[[], Any] = _default_profiler_factory,
) -> Optional[Dict[str, Any]]:
    """Load or compute the local hardware profile as a dict."""
    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR

    # 1) explicit env override
    env_path = os.environ.get("PRSM_HARDWARE_PROFILE_FILE", "").strip()
    if env_path:
        env_file = Path(env_path)
        if not env_file.exists():
            logger.warning(
                "PRSM_HARDWARE_PROFILE_FILE=%s does not exist; peer "
                "will not advertise hardware_profile.", env_path,
            )
            return None
        try:
            data = json.loads(env_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "PRSM_HARDWARE_PROFILE_FILE=%s failed to parse: %s; "
                "peer will not advertise hardware_profile.",
                env_path, exc,
            )
            return None
        if not isinstance(data, dict):
            logger.warning(
                "PRSM_HARDWARE_PROFILE_FILE=%s did not contain a JSON "
                "object; peer will not advertise hardware_profile.",
                env_path,
            )
            return None
        _merge_operator_address(data)
        _merge_operator_delegation(data)
        _merge_hardware_overrides(data)
        _merge_attestation(data)
        return data

    # 2) on-disk cache
    cache_file = cache_dir / _CACHE_FILENAME
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if isinstance(data, dict):
                _merge_operator_address(data)
                _merge_operator_delegation(data)
                _merge_hardware_overrides(data)
                _merge_attestation(data)
                return data
            logger.warning(
                "hardware_profile cache %s top-level is not a dict; "
                "recomputing.", cache_file,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "hardware_profile cache %s unreadable (%s); recomputing.",
                cache_file, exc,
            )

    # 3) fresh compute
    try:
        profiler = profiler_factory()
        profile = profiler.detect()
        data = profile.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Hardware profiler raised: %s; peer will not advertise "
            "hardware_profile.", exc,
        )
        return None

    # Cache for next start (non-fatal on failure)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
    except (OSError, PermissionError) as exc:
        logger.debug(
            "Could not write hardware_profile cache to %s: %s "
            "(non-fatal).", cache_file, exc,
        )

    _merge_operator_address(data)
    _merge_operator_delegation(data)
    _merge_hardware_overrides(data)
    _merge_attestation(data)
    return data


def _merge_hardware_overrides(data: Dict[str, Any]) -> None:
    """Sprint 843 — propagate producer-side hw overrides into the
    advertised profile so the sp838 relay carries them to other
    consumers.

    Pre-843 the three override envs (TFLOPS_FP16, MEMORY_GB,
    LAYER_CAPACITY) were consumer-side only — setting them on
    a droplet only affected what THAT droplet saw in ITS pool
    view, not what a remote consumer (e.g., a Mac joining via
    bootstrap relay) saw for THAT droplet. Multi-host live test
    2026-05-25 (post-sp838 fleet deploy) confirmed the gap: Mac
    saw 1.92GB raw memory + computed cap=1 for both 2GB droplets
    despite each droplet's local env having LAYER_CAPACITY_OVERRIDE=3.

    Sprint 843 fix: producer writes the env-resolved overrides
    into the hw_profile dict as explicit fields
    (``tflops_fp16_override``, ``memory_gb_override``,
    ``layer_capacity_override``). Consumer reads per-peer
    fields first, falls back to its own env if absent. The
    consumer-side env behavior becomes a coarse default for
    peers that don't advertise an explicit override.

    Wire-format compatibility: missing/invalid env → no key
    written → pre-843 relay shape preserved.
    """
    for env_key, dict_key, caster in (
        ("PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE",
         "tflops_fp16_override", float),
        ("PRSM_PARALLAX_MEMORY_GB_OVERRIDE",
         "memory_gb_override", float),
        ("PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE",
         "layer_capacity_override", int),
    ):
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        try:
            value = caster(raw)
        except (ValueError, TypeError):
            logger.debug(
                "Sprint 843 — %s=%r not parseable as %s; skipping",
                env_key, raw, caster.__name__,
            )
            continue
        # Reject non-positive — same posture as the consumer-side
        # validators in dht_backed_pool_provider.
        if value <= 0:
            continue
        data[dict_key] = value


def _merge_operator_address(data: Dict[str, Any]) -> None:
    """Sprint 690 F31 fix piece 1 — merge PRSM_OPERATOR_ADDRESS
    into the loaded profile so the DHT pool provider's
    operator_address path (sprint 683) gets a real on-chain
    address to look up stake against.

    Validates 0x-prefixed 42-char hex form (Ethereum). Garbage →
    skip + warn; absent env → no field added.

    Called both on the explicit env-pin path AND on the cache/
    compute paths so EVERY return shape gets the merge.
    """
    addr = os.environ.get("PRSM_OPERATOR_ADDRESS", "").strip()
    if not addr:
        return
    if (
        not addr.startswith("0x")
        or len(addr) != 42
        or not all(c in "0123456789abcdefABCDEF" for c in addr[2:])
    ):
        logger.warning(
            "PRSM_OPERATOR_ADDRESS=%r is not a valid 0x-prefixed "
            "42-char hex Ethereum address; peer will NOT advertise "
            "operator_address (stake-eligibility will fail under "
            "enforced mode).", addr,
        )
        return
    data["operator_address"] = addr


def _merge_operator_delegation(data: Dict[str, Any]) -> None:
    """Sprint 797 — merge the operator's EIP-191 delegation into
    the loaded hardware_profile so peers can verify
    operator_address (sprint 788) instead of trusting the bare
    claim.

    Resolution order (first wins):
      1. PRSM_OPERATOR_DELEGATION env (raw JSON string)
      2. PRSM_OPERATOR_DELEGATION_FILE env (path to JSON)
      3. ~/.prsm/operator_delegation.json (default location;
         matches the `wallet devices add --write` default).

    Malformed JSON / missing file / read failure → skip silently
    (warn at debug level). The peer-verify side (sprint 788)
    then treats this node as effectively unstaked, which is the
    fail-safe behavior — no peer is BROKEN by a missing
    delegation, they just lose stake-tier privilege.
    """
    blob: Optional[Dict[str, Any]] = None

    # 1) Env raw JSON
    raw = os.environ.get("PRSM_OPERATOR_DELEGATION") or ""
    raw = raw.strip()
    if raw:
        try:
            blob = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "PRSM_OPERATOR_DELEGATION env is not valid JSON; "
                "peer will not advertise operator_delegation (peers "
                "will reject operator_address claim): %s", exc,
            )
            return

    # 2) Env file path
    if blob is None:
        file_path = (os.environ.get(
            "PRSM_OPERATOR_DELEGATION_FILE",
        ) or "").strip()
        if file_path:
            try:
                from pathlib import Path
                blob = json.loads(Path(file_path).read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug(
                    "PRSM_OPERATOR_DELEGATION_FILE=%s unreadable "
                    "(%s); skipping.", file_path, exc,
                )
                return

    # 3) Default path ~/.prsm/operator_delegation.json
    if blob is None:
        try:
            from pathlib import Path
            default_path = (
                Path.home() / ".prsm" / "operator_delegation.json"
            )
            if default_path.exists():
                blob = json.loads(default_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug(
                "default operator_delegation.json read failed "
                "(%s); skipping.", exc,
            )
            return

    if blob is None:
        return

    if not isinstance(blob, dict):
        logger.warning(
            "operator_delegation source did not yield a JSON "
            "object; skipping."
        )
        return

    data["operator_delegation"] = blob


def load_tee_attestation_blob() -> Optional[bytes]:
    """Sprint 1114 — load THIS node's raw TEE attestation quote bytes from the operator's
    environment, for the dispatch-time TEE-policy gate (api._enforce_tee_policy reads
    node._tee_node_attestation_blob). Resolution order (first wins):
      1. PRSM_TEE_ATTESTATION_B64  (the quote, base64-encoded)
      2. PRSM_TEE_ATTESTATION_FILE (path to the raw binary quote)
    Returns the raw bytes, or None when unset/unreadable/oversized (the node then stays
    tier-none = ineligible for confidential work — the fail-safe default). Capped at 64
    KiB (a genuine SGX/TDX/SEV quote is ≤ ~10 KB; mirrors the advertise cap +
    trust_adapter._MAX_ATTESTATION_BYTES)."""
    import base64
    import os

    _MAX = 64 * 1024
    b64 = (os.environ.get("PRSM_TEE_ATTESTATION_B64") or "").strip()
    if b64:
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PRSM_TEE_ATTESTATION_B64 is not valid base64 (%s); node "
                           "stays tier-none.", exc)
            return None
        if len(raw) > _MAX:
            logger.warning("PRSM_TEE_ATTESTATION_B64 is %d bytes (> %d cap) — not a real "
                           "TEE quote; ignoring.", len(raw), _MAX)
            return None
        return raw
    file_path = (os.environ.get("PRSM_TEE_ATTESTATION_FILE") or "").strip()
    if file_path:
        try:
            from pathlib import Path
            raw = Path(file_path).read_bytes()
        except OSError as exc:
            logger.debug("PRSM_TEE_ATTESTATION_FILE=%s unreadable (%s); skipping.",
                         file_path, exc)
            return None
        if raw and len(raw) <= _MAX:
            return raw
        if raw:
            logger.warning("PRSM_TEE_ATTESTATION_FILE is %d bytes (> %d cap) — ignoring.",
                           len(raw), _MAX)
    return None


def _merge_attestation(data: Dict[str, Any]) -> None:
    """Sprint 1083 — merge this node's TEE attestation quote into the advertised
    hardware_profile so a consumer can CRYPTOGRAPHICALLY verify it (sp1083
    verified_tier_attestation) before routing confidential Tier B/C work here, instead
    of trusting a self-advertised tier string.

    Resolution order (first wins):
      1. PRSM_TEE_ATTESTATION_B64 env (the quote, base64-encoded)
      2. PRSM_TEE_ATTESTATION_FILE env (path to the raw binary quote)

    The advertised ``attestation`` field is ALWAYS base64. Missing / unreadable → skip
    silently (the node stays tier-none = ineligible for confidential work, the fail-safe
    default). A non-TEE node has no quote and simply advertises nothing here — unchanged.
    """
    import base64

    # sp1114 — reuse the single loader so the advertised attestation and the
    # dispatch-gate blob can never diverge (same source, caps, validation).
    raw = load_tee_attestation_blob()
    if raw:
        data["attestation"] = base64.b64encode(raw).decode()
