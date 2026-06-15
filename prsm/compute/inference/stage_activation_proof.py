"""Per-stage signed activation-hash chain for the production §7 inference receipt.

Domain-03 review F1/F2: the production ``InferenceReceipt`` commits to the OUTPUT
(output_hash), the INPUT prompt (prompt_hash, sp1099), and the orchestrator's UNSIGNED
``topology_assignment`` — but NOT to a per-stage, per-node-signed record of which node
ran which layer slice and what activations flowed through it. So in a multi-stage swarm a
malicious head node can misattribute work (claim a different topology) or substitute the
computation, and the receipt still verifies.

This module is the primitive that closes that gap. Each worker, when it runs its layer
slice, signs a compact ``StageActivationProof`` binding:

    (request_id, stage_index, stage_node_id, input_activation_hash, output_activation_hash)

Those signed proofs are collected by the orchestrator into a ``StageActivationChain`` that
is carried in — and cryptographically bound into — the ``InferenceReceipt`` (brick 1 wires
the receptacle; brick 2 has workers produce the proofs; brick 3 threads them; brick 4
verifies the per-stage signatures + the chain against the on-chain anchor and DERIVES the
topology from the signed stages instead of trusting the orchestrator's assertion).

The chain's integrity is twofold:
  * CONTINUITY (no anchor needed, ``verify_continuity`` here): stage 0's input hash is the
    prompt hash, each stage's output hash is the next stage's input hash, and the last
    stage's output hash is the receipt's output hash. A spliced/reordered/substituted
    intermediate activation breaks the chain.
  * AUTHENTICITY (needs the anchor, brick 4): each proof's signature verifies under the
    claimed stage_node_id's published key — so the node_id in each link is PROVEN, and the
    topology is derivable from the signed stages.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Domain separator + version for the per-stage signed payload. Bumping this is a
# wire-format break — verifiers and signers must agree byte-for-byte.
_STAGE_PROOF_DOMAIN = "prsm-stage-activation-proof-v1"


@dataclass(frozen=True)
class StageActivationProof:
    """One worker's signed attestation that it ran ``stage_index`` for ``request_id``,
    receiving an input whose sha256 is ``input_activation_hash`` and producing an output
    whose sha256 is ``output_activation_hash``."""

    stage_index: int
    stage_node_id: str
    input_activation_hash: str   # sha256 hex of the input activation to this stage
    output_activation_hash: str  # sha256 hex of the output activation of this stage
    stage_signature_b64: str = ""  # Ed25519 sig (by stage_node_id) over signing_bytes()

    def signing_bytes(self, request_id: str) -> bytes:
        """Canonical bytes the WORKER signs. Binds the stage to the specific request +
        its node_id + its in/out activation hashes, so a signed proof cannot be replayed
        onto a different request, stage, node, or activation pair. Excludes the signature
        itself (would be circular). Field order is fixed; do not reorder without bumping
        ``_STAGE_PROOF_DOMAIN``."""
        parts = [
            _STAGE_PROOF_DOMAIN,
            str(request_id),
            str(int(self.stage_index)),
            str(self.stage_node_id),
            str(self.input_activation_hash),
            str(self.output_activation_hash),
        ]
        return "\n".join(parts).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_index": int(self.stage_index),
            "stage_node_id": str(self.stage_node_id),
            "input_activation_hash": str(self.input_activation_hash),
            "output_activation_hash": str(self.output_activation_hash),
            "stage_signature_b64": str(self.stage_signature_b64),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageActivationProof":
        return cls(
            stage_index=int(d["stage_index"]),
            stage_node_id=str(d["stage_node_id"]),
            input_activation_hash=str(d["input_activation_hash"]),
            output_activation_hash=str(d["output_activation_hash"]),
            stage_signature_b64=str(d.get("stage_signature_b64", "")),
        )


@dataclass(frozen=True)
class StageActivationChain:
    """The ordered set of per-stage signed proofs for one inference, carried in the
    ``InferenceReceipt``. Bound into the receipt's signing payload via ``stable_hash``."""

    request_id: str
    proofs: List[StageActivationProof]

    def _sorted_proofs(self) -> List[StageActivationProof]:
        return sorted(self.proofs, key=lambda p: int(p.stage_index))

    def stable_hash(self) -> str:
        """Canonical sha256 hex over the request_id + the stage-sorted proofs (incl. each
        proof's signature). Bound into ``InferenceReceipt.signing_payload`` so tampering
        any link — a hash, a node_id, a signature, or the stage order — flips the receipt
        signature. Mirrors the sprint-297 topology/partial_completion encoding."""
        canon = json.dumps(
            {
                "request_id": str(self.request_id),
                "proofs": [p.to_dict() for p in self._sorted_proofs()],
            },
            sort_keys=True,
        )
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": str(self.request_id),
            "proofs": [p.to_dict() for p in self._sorted_proofs()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageActivationChain":
        return cls(
            request_id=str(d.get("request_id", "")),
            proofs=[
                StageActivationProof.from_dict(p)
                for p in d.get("proofs", [])
            ],
        )

    def verify_internal_links(self) -> Tuple[bool, str]:
        """Pure (no-anchor, no-endpoint) check of the chain's INTERNAL structure:
          1. >= 1 stage,
          2. stage indices are 0..N-1 with no gaps/dupes,
          3. each stage's output hash == the next stage's input hash.
        This is the stage-to-stage splice/reorder/substitution defense — it needs neither
        the anchor (signatures) nor the prompt/output endpoints (which involve the head's
        prompt-encode / output-decode framing). The per-stage SIGNATURE authenticity and
        the prompt/output endpoint tie are checked separately."""
        stages = self._sorted_proofs()
        if not stages:
            return False, "empty activation chain"
        for expected, p in enumerate(stages):
            if int(p.stage_index) != expected:
                return False, (
                    f"stage indices not contiguous from 0 (got {p.stage_index} "
                    f"at position {expected})"
                )
        for k in range(len(stages) - 1):
            if stages[k].output_activation_hash != stages[k + 1].input_activation_hash:
                return False, f"chain broken between stage {k} and {k + 1}"
        return True, "internal activation links intact"

    def verify_continuity(
        self, *, prompt_hash_hex: str, output_hash_hex: str,
    ) -> Tuple[bool, str]:
        """Full continuity check: the internal links PLUS the prompt/output endpoints
        (stage 0's input hash == prompt_hash_hex, last stage's output hash ==
        output_hash_hex). Use this when the caller knows the chain-entry / chain-exit
        ACTIVATION hashes (A0 / AN — what the orchestrator encoded the prompt to / the
        final activation it decoded). For an endpoint-agnostic structural check use
        verify_internal_links()."""
        ok, why = self.verify_internal_links()
        if not ok:
            return False, why
        stages = self._sorted_proofs()
        if stages[0].input_activation_hash != prompt_hash_hex:
            return False, "chain broken at prompt entry (stage 0 input != prompt_hash)"
        if stages[-1].output_activation_hash != output_hash_hex:
            return False, "chain broken at output exit (last stage output != output_hash)"
        return True, "activation chain continuity intact"

    def topology_node_ids(self) -> List[str]:
        """The per-stage node_ids in stage order — the topology DERIVED from the signed
        stages (brick 4 cross-checks this against the receipt's topology_assignment so the
        orchestrator can't assert a topology the signed stages don't support)."""
        return [p.stage_node_id for p in self._sorted_proofs()]


def activation_hash(blob: bytes) -> str:
    """The canonical sha256-hex of an activation blob, used on both the worker (when it
    signs its proof) and the verifier. Centralized so the two never diverge."""
    return hashlib.sha256(bytes(blob)).hexdigest()


# ── Verification (sp1108, brick 2) ────────────────────────────────────────────────
# These are the AUTHENTICITY half of the chain (the CONTINUITY half is
# StageActivationChain.verify_continuity above). They need the on-chain anchor to
# resolve each stage node_id's published key, so they live as free functions rather than
# methods (keeping the dataclasses dependency-light for models.py to import).


def verify_stage_proof(proof: "StageActivationProof", request_id: str, *, anchor: Any) -> bool:
    """True iff ``proof.stage_signature_b64`` is a valid Ed25519 signature, by the key the
    anchor publishes for ``proof.stage_node_id``, over the proof's canonical signing bytes
    for ``request_id``. Fail-closed: a missing key, empty signature, or any verification
    error returns False."""
    from prsm.node.identity import verify_signature

    if not proof.stage_signature_b64:
        return False
    pubkey_b64 = anchor.lookup(proof.stage_node_id)
    if not pubkey_b64:
        return False
    try:
        return verify_signature(
            pubkey_b64, proof.signing_bytes(request_id), proof.stage_signature_b64,
        )
    except Exception:  # noqa: BLE001 — verification must never raise into the caller
        return False


def verify_stage_activation_chain(
    chain: "StageActivationChain", *,
    prompt_hash_hex: str, output_hash_hex: str, anchor: Any,
    expected_request_id: Any = None,
) -> Tuple[bool, str]:
    """Full chain verification: CONTINUITY (the activations actually link prompt→output
    with no spliced intermediate) AND AUTHENTICITY (every stage's link is signed by the
    node the anchor knows). When ``expected_request_id`` is supplied, also binds the
    chain to THAT request (rejecting a valid chain spliced in from a different inference).
    Returns (ok, diagnostic). A True result means: the receipt's prompt produced its
    output through exactly these stages, each run by the node that cryptographically
    signed for it — the §7 promise, end to end."""
    if (
        expected_request_id is not None
        and str(chain.request_id) != str(expected_request_id)
    ):
        return False, (
            f"chain request_id {chain.request_id!r} does not match the receipt's "
            f"{expected_request_id!r} (chain spliced from a different inference)"
        )
    ok, why = chain.verify_continuity(
        prompt_hash_hex=prompt_hash_hex, output_hash_hex=output_hash_hex,
    )
    if not ok:
        return False, why
    for p in sorted(chain.proofs, key=lambda x: int(x.stage_index)):
        if not verify_stage_proof(p, chain.request_id, anchor=anchor):
            return False, (
                f"stage {p.stage_index} signature invalid or unknown node "
                f"{p.stage_node_id}"
            )
    return True, "stage activation chain verified (continuity + per-stage signatures)"


class _DictAnchor:
    """Minimal anchor (``lookup(node_id) -> pubkey_b64``) backed by a caller-supplied
    dict — lets the verify endpoint do per-stage signature checks self-contained (the
    caller supplies the stage keys, mirroring how it supplies the settler public key)."""

    def __init__(self, mapping: Dict[str, str]) -> None:
        self._m = mapping or {}

    def lookup(self, node_id: str) -> Any:
        return self._m.get(node_id)


def summarize_stage_activation_chain(
    chain: "StageActivationChain", *,
    stage_public_keys: Any = None,
    topology_assignment: Any = None,
    expected_request_id: Any = None,
) -> Dict[str, Any]:
    """Build a JSON-able verification summary of a receipt's activation chain for the
    /compute/receipt/verify response. Always reports the internal-link structure +
    derived topology (no keys needed). When ``stage_public_keys`` ({node_id: b64}) is
    supplied, also verifies every per-stage signature (authenticity); when
    ``topology_assignment`` is supplied, cross-checks it against the signed stages (F2);
    when ``expected_request_id`` (the RECEIPT's request_id) is supplied, binds the chain
    to THIS receipt — closing a cross-inference splice where a malicious head presents a
    valid chain from a DIFFERENT inference (its per-stage signatures + internal links +
    topology all verify, because they're genuine — for the OTHER request). Fail-closed:
    any signature that can't be verified makes signatures_valid False."""
    links_ok, links_detail = chain.verify_internal_links()
    out: Dict[str, Any] = {
        "present": True,
        "stage_count": len(chain.proofs),
        "node_ids": chain.topology_node_ids(),
        "internal_links_ok": links_ok,
        "internal_links_detail": links_detail,
    }
    if expected_request_id is not None:
        # The proofs are each signed over chain.request_id; binding chain.request_id to
        # the receipt's request_id is what ties those genuine signatures to THIS
        # inference (without it, a valid chain from another request would pass).
        out["request_id_bound"] = (
            str(chain.request_id) == str(expected_request_id)
        )
    if stage_public_keys:
        anchor = _DictAnchor(dict(stage_public_keys))
        all_ok = True
        for p in chain.proofs:
            if not verify_stage_proof(p, chain.request_id, anchor=anchor):
                all_ok = False
                break
        out["signatures_checked"] = True
        out["signatures_valid"] = all_ok
    else:
        out["signatures_checked"] = False
    if topology_assignment is not None:
        topo_ok, topo_detail = chain_supports_topology(chain, topology_assignment)
        out["topology_consistent"] = topo_ok
        out["topology_detail"] = topo_detail
    return out


def chain_supports_topology(
    chain: "StageActivationChain", topology_assignment: Any,
) -> Tuple[bool, str]:
    """Cross-check that the orchestrator's asserted ``topology_assignment`` is CONSISTENT
    with the cryptographically-proven stages (F2): for each stage, the node that signed
    must be one the topology assigns to that stage. This is what lets a verifier stop
    trusting the orchestrator's unsigned topology and instead derive it from the signed
    stages. ``topology_assignment`` is the sprint-296 TopologyAssignment
    (positions: {(stage_idx, slot_idx): node_id}); None means nothing to cross-check."""
    if topology_assignment is None:
        return True, "no topology to cross-check"
    positions = getattr(topology_assignment, "positions", None)
    if not isinstance(positions, dict):
        return False, "topology_assignment has no positions mapping"
    # node_ids the topology assigns to each stage (across all slots)
    by_stage: Dict[int, set] = {}
    for (stage_idx, _slot), node in positions.items():
        by_stage.setdefault(int(stage_idx), set()).add(node)
    for p in chain.proofs:
        allowed = by_stage.get(int(p.stage_index))
        if not allowed or p.stage_node_id not in allowed:
            return False, (
                f"stage {p.stage_index} was signed by {p.stage_node_id}, which the "
                f"asserted topology does not assign to that stage"
            )
    return True, "topology is consistent with the signed stages"
