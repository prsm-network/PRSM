"""Standalone PRSM InferenceReceipt verifier.

Sprint 703 — a single-file Python script that verifies a PRSM
inference receipt without cloning the PRSM repo or running any
PRSM daemon. Lets a journalist, auditor, or skeptical engineer
independently confirm that:

  1. The settler's pubkey is registered on the on-chain
     `PublisherKeyAnchor` contract (Base mainnet by default).
  2. The settler_signature in the receipt is a valid Ed25519
     signature of the canonical signing bytes against that pubkey.
  3. The multi-stage tee_attestation envelope parses cleanly.
  4. The activation_noise_trace (when present) commits to the
     epsilon vector that the receipt claims.

USAGE
=====

    pip install web3 eth-account cryptography
    python3 verify_prsm_receipt.py receipt.json
        [--anchor 0xd811ad9986f44f404b0fd992168a7cc76206df03]
        [--rpc https://mainnet.base.org]

EXIT CODES
==========

    0 — receipt verifies cleanly (signature valid + anchor lookup
        matches + attestation envelope parses)
    1 — verification failed (any of the checks above)
    2 — invalid input (file not found, JSON malformed, etc.)

NO PRSM IMPORTS
===============

This script intentionally has zero imports from the `prsm`
package. It re-implements the canonical signing-payload bytes
from the receipt schema documented in PRSM_Vision.md §7 + the
audit-readiness doc. If PRSM's repo disappeared, an external
party with only (a) this file, (b) the receipt JSON, and (c)
any Base mainnet RPC URL would still verify the receipt.

Dependencies are minimal and uncontroversial:
  - web3 (Ethereum RPC client)
  - eth-account (used by web3)
  - cryptography (Ed25519 verification)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path


DEFAULT_ANCHOR_ADDRESS = "0xd811ad9986f44f404b0fd992168a7cc76206df03"
DEFAULT_BASE_RPC_URL = "https://mainnet.base.org"

PUBLISHER_KEY_ANCHOR_LOOKUP_ABI = [
    {
        "name": "publisherKeys",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "", "type": "bytes16"}],
        "outputs": [{"name": "", "type": "bytes"}],
    },
]


def _topology_stable_hash(positions: list) -> str:
    """Re-implement TopologyAssignment.stable_hash from
    prsm/compute/inference/topology_rotation.py:76. Positions
    are a list of [stage, slot, node_id] triples."""
    sorted_pos = sorted(
        positions, key=lambda x: (int(x[0]), int(x[1])),
    )
    canonical = json.dumps(sorted_pos, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_signing_payload(receipt: dict) -> bytes:
    """Re-implement InferenceReceipt.signing_payload from
    prsm/compute/inference/models.py:252.

    Order MUST match exactly — any change breaks signature
    verification for all in-flight receipts.
    """
    parts = [
        receipt["job_id"],
        receipt["request_id"],
        receipt["model_id"],
        receipt["content_tier"],
        receipt["privacy_tier"],
        f"{float(receipt['epsilon_spent']):.10f}",
        receipt["tee_type"],
        receipt["tee_attestation"],  # already hex-string in JSON
        receipt["output_hash"],  # already hex-string
        f"{float(receipt['duration_seconds']):.6f}",
        str(receipt["cost_ftns"]),
        receipt["settler_node_id"],
    ]
    if receipt.get("streamed_output"):
        parts.append("streamed_output:true")
    trace = receipt.get("activation_noise_trace")
    if trace is not None:
        trace_canon = json.dumps(trace, sort_keys=True)
        trace_hash = hashlib.sha256(
            trace_canon.encode("utf-8"),
        ).hexdigest()
        parts.append(f"activation_noise_trace:{trace_hash}")
    topo = receipt.get("topology_assignment")
    if topo is not None:
        topo_hash = _topology_stable_hash(topo["positions"])
        parts.append(f"topology_assignment:{topo_hash}")
    # sprint 777 — partial_completion folds into the payload via a canonical
    # sha256 over the sorted-key JSON of its dict (PartialCompletionInfo.
    # stable_hash). Kept in sync here so the standalone verifier can validate
    # partial-completion receipts (previously diverged → false-invalid).
    partial = receipt.get("partial_completion")
    if partial is not None:
        partial_canon = json.dumps(partial, sort_keys=True)
        partial_hash = hashlib.sha256(
            partial_canon.encode("utf-8"),
        ).hexdigest()
        parts.append(f"partial_completion:{partial_hash}")
    # sp1099 — prompt_hash binds the input prompt; trailing conditional append,
    # already a hex string in the JSON.
    prompt_hash = receipt.get("prompt_hash")
    if prompt_hash is not None:
        parts.append(f"prompt_hash:{prompt_hash}")
    return "\n".join(parts).encode("utf-8")


def _node_id_to_bytes16(node_id_hex: str) -> bytes:
    """node_id is 32 hex chars (16 bytes). Convert to bytes16."""
    if len(node_id_hex) != 32:
        raise ValueError(
            f"node_id must be 32 hex chars; got {len(node_id_hex)}"
        )
    return bytes.fromhex(node_id_hex)


def _anchor_lookup(
    node_id: str, anchor_address: str, rpc_url: str,
) -> bytes:
    """Look up the settler's published Ed25519 pubkey on the
    on-chain PublisherKeyAnchor."""
    try:
        from web3 import Web3
    except ImportError:
        raise SystemExit(
            "web3 not installed. Run: pip install web3 eth-account"
        )

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise RuntimeError(
            f"web3 cannot reach {rpc_url}"
        )
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(anchor_address),
        abi=PUBLISHER_KEY_ANCHOR_LOOKUP_ABI,
    )
    bytes16 = _node_id_to_bytes16(node_id)
    raw = contract.functions.publisherKeys(bytes16).call()
    return bytes(raw)


def _verify_ed25519(
    pubkey_bytes: bytes, message: bytes, signature: bytes,
) -> bool:
    """Verify Ed25519 signature using the `cryptography` library."""
    try:
        from cryptography.hazmat.primitives.asymmetric import (
            ed25519,
        )
        from cryptography.exceptions import InvalidSignature
    except ImportError:
        raise SystemExit(
            "cryptography not installed. Run: pip install cryptography"
        )
    try:
        pubkey = ed25519.Ed25519PublicKey.from_public_bytes(
            pubkey_bytes,
        )
        pubkey.verify(signature, message)
        return True
    except InvalidSignature:
        return False
    except Exception:  # noqa: BLE001 — defensive
        return False


def _decode_attestation_envelope(att_hex: str) -> dict:
    """Decode the multi-stage attestation envelope from the receipt.
    The envelope starts with the magic prefix `PRSM-MS-ATT-V1:`
    followed by a JSON blob listing per-stage attestations.

    Returns the parsed dict (with `stages` and `version` keys) or
    raises ValueError if the prefix is missing or the JSON is
    malformed.
    """
    blob = bytes.fromhex(att_hex)
    PREFIX = b"PRSM-MS-ATT-V1:"
    if not blob.startswith(PREFIX):
        raise ValueError(
            f"attestation envelope missing PRSM-MS-ATT-V1: prefix "
            f"(got first 16 bytes {blob[:16]!r})"
        )
    payload = blob[len(PREFIX):]
    return json.loads(payload.decode("utf-8"))


def _check_activation_noise_trace_integrity(trace: dict) -> list:
    """Sanity-check the activation_noise_trace fields. Returns a
    list of human-readable findings (empty list = clean)."""
    findings = []
    per_stage = trace.get("per_stage_epsilon")
    total = trace.get("total_epsilon_spent")
    if per_stage is None or total is None:
        findings.append("trace missing per_stage_epsilon or total_epsilon_spent")
        return findings
    if not isinstance(per_stage, list) or not all(
        isinstance(x, (int, float)) for x in per_stage
    ):
        findings.append("per_stage_epsilon must be a list of numbers")
        return findings
    summed = sum(per_stage)
    # Sum should match total (up to float fuzz).
    if abs(summed - total) > 1e-6:
        findings.append(
            f"per_stage_epsilon sums to {summed}, but "
            f"total_epsilon_spent claims {total}"
        )
    return findings


def verify(
    receipt: dict,
    anchor_address: str = DEFAULT_ANCHOR_ADDRESS,
    rpc_url: str = DEFAULT_BASE_RPC_URL,
) -> dict:
    """Verify a PRSM receipt. Returns a structured result dict."""
    result = {
        "valid": False,
        "anchor_lookup": None,
        "signature_valid": None,
        "attestation_envelope": None,
        "noise_trace": None,
        "findings": [],
    }

    settler_node_id = receipt.get("settler_node_id", "")
    settler_signature_hex = receipt.get("settler_signature", "")
    if not settler_node_id or not settler_signature_hex:
        result["findings"].append(
            "receipt missing settler_node_id or settler_signature"
        )
        return result

    try:
        pubkey_b64 = _anchor_lookup(
            settler_node_id, anchor_address, rpc_url,
        )
    except Exception as exc:  # noqa: BLE001
        result["findings"].append(f"anchor lookup failed: {exc}")
        return result

    if not pubkey_b64:
        result["findings"].append(
            f"settler_node_id {settler_node_id} not registered on anchor"
        )
        return result

    # PublisherKeyAnchor stores raw pubkey bytes (32 bytes Ed25519)
    if len(pubkey_b64) != 32:
        # Some encodings return base64-encoded; try decode
        try:
            pubkey_bytes = base64.b64decode(pubkey_b64)
        except Exception:
            pubkey_bytes = pubkey_b64
    else:
        pubkey_bytes = bytes(pubkey_b64)

    result["anchor_lookup"] = base64.b64encode(pubkey_bytes).decode()

    signing_bytes = _build_signing_payload(receipt)
    signature_bytes = bytes.fromhex(settler_signature_hex)

    result["signature_valid"] = _verify_ed25519(
        pubkey_bytes, signing_bytes, signature_bytes,
    )
    if not result["signature_valid"]:
        result["findings"].append(
            "settler_signature does NOT verify against the anchor-"
            "registered pubkey — receipt was tampered with OR the "
            "settler used a different key than the one registered"
        )

    try:
        envelope = _decode_attestation_envelope(
            receipt.get("tee_attestation", ""),
        )
        result["attestation_envelope"] = {
            "version": envelope.get("version"),
            "stage_count": len(envelope.get("stages", [])),
            "stages": [
                {
                    "stage_index": s.get("stage_index"),
                    "stage_node_id": s.get("stage_node_id"),
                    "tee_type": s.get("tee_type"),
                }
                for s in envelope.get("stages", [])
            ],
        }
    except Exception as exc:  # noqa: BLE001
        result["findings"].append(
            f"attestation envelope decode failed: {exc}"
        )

    trace = receipt.get("activation_noise_trace")
    if trace is not None:
        findings = _check_activation_noise_trace_integrity(trace)
        result["noise_trace"] = {
            "tier": trace.get("tier"),
            "stage_count": trace.get("stage_count"),
            "total_epsilon_spent": trace.get("total_epsilon_spent"),
            "integrity_findings": findings,
        }
        if findings:
            result["findings"].extend(findings)

    result["valid"] = (
        result["signature_valid"] is True
        and not result["findings"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone PRSM InferenceReceipt verifier. "
            "Independent of the PRSM repo; needs only web3 + "
            "cryptography + a receipt JSON file."
        ),
    )
    parser.add_argument(
        "receipt",
        help="Path to receipt JSON file (the `receipt` field from a "
             "successful /compute/inference response)",
    )
    parser.add_argument(
        "--anchor", default=DEFAULT_ANCHOR_ADDRESS,
        help=f"PublisherKeyAnchor contract address (default: {DEFAULT_ANCHOR_ADDRESS})",
    )
    parser.add_argument(
        "--rpc", default=DEFAULT_BASE_RPC_URL,
        help=f"Base RPC URL (default: {DEFAULT_BASE_RPC_URL})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Print only `valid: true/false` (suitable for CI)",
    )
    args = parser.parse_args()

    receipt_path = Path(args.receipt)
    if not receipt_path.exists():
        print(f"ERROR: {receipt_path} not found", file=sys.stderr)
        return 2
    try:
        receipt = json.loads(receipt_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON: {exc}", file=sys.stderr)
        return 2

    # Allow either the bare receipt or the full /compute/inference
    # response (which wraps it in {"success": ..., "receipt": ...}).
    if "receipt" in receipt and isinstance(receipt["receipt"], dict):
        receipt = receipt["receipt"]

    result = verify(
        receipt, anchor_address=args.anchor, rpc_url=args.rpc,
    )

    if args.quiet:
        print("valid:", result["valid"])
        return 0 if result["valid"] else 1

    print(json.dumps(result, indent=2))
    print()
    if result["valid"]:
        print("✓ VALID — receipt verifies cleanly")
    else:
        print("✗ INVALID — see findings above")
        for f in result["findings"]:
            print(f"  - {f}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
