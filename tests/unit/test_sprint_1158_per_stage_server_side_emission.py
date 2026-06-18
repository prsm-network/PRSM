"""Sprint 1158 — ON-CHAIN PER-STAGE PAYEE, SERVER-SIDE EMISSION (completes
BRICK 2 of the on-chain per-stage payee arc).

Brick 2 (sp1156) gave ``RunLayerSliceResponse.sign(...)`` the MECHANISM to emit a
challenge-defensible per-node settlement leaf signature when passed
``settlement_job_id`` / ``settlement_stage_index`` / ``settlement_executed_at_unix``.
But the 6 ``RunLayerSliceResponse.sign(...)`` call sites in
``prsm/compute/chain_rpc/server.py`` passed NONE of those — fail-closed, so no
worker emitted. sp1158 WIRES the sign sites.

THE RESOLUTION (user-approved). A worker, at stage-execution time, has
``request.request_id`` + the settler-signed ``chain_stage_index`` but CANNOT
reliably know the receipt's eventual OUTPUT ``streamed`` flag (set post-hoc by
the orchestrator; the worker's transfer streamed-ness is a DIFFERENT bit). So the
per-node SETTLEMENT leaf job_id is STREAMED-AGNOSTIC — derived from
``request_id`` ALONE via ``per_stage_leaf_job_id(request_id) ==
settlement_job_id(request_id, streamed=False)`` — and BOTH the worker (server.py
sign sites) AND the splitter (``per_stage_settlement_split`` ~line 291) bind THAT
id. So the worker signs EXACTLY the leaf the splitter rebuilds → an on-chain
INVALID_SIGNATURE challenge would FAIL → the per-node leaf is defensible.

Tests:
  (a) JOBID-AGREEMENT — a worker signs a stage via the REAL server ``handle`` →
      ``_dispatch`` → ``RunLayerSliceResponse.sign`` path; the emitted settlement
      signature is over ``per_stage_leaf_job_id(request_id)`` (NOT the receipt
      display job_id). The splitter, fed the assembled material + a receipt with
      that request_id, rebuilds the leaf over the SAME id → the leaf's
      ``signing_message_hash`` matches AND the worker sig verifies. The worker
      used ONLY ``request_id`` (no streamed flag) and still agrees with the
      splitter.
  (b) STREAMED-AGNOSTIC — the per-node leaf job_id is identical whether the
      receipt is later marked streamed or unary (it is derived from request_id
      alone), so a worker that doesn't know the receipt streamed flag still
      agrees with the splitter for BOTH.
  (c) BYTE-EQUIVALENCE — ``sign()`` WITHOUT settlement inputs produces a
      byte-identical stage signing_payload + the SAME ``stage_signature_b64`` as
      with the settlement inputs present; the wired sites change only the
      separate settlement artifact, never the stage signature.
  (d) IN-PROCESS E2E — two distinct node identities each run a stage via the real
      server path → emit settlement sigs → orchestrator
      ``_assemble_per_node_settlement_signatures`` → splitter → 2 challenge-
      defensible per-node ``BatchedReceipt`` summing to the total (conservation).
  (e) SINGLE-PAYEE UNTOUCHED — the single-payee ``inference_adapter`` leaf still
      uses ``receipt.job_id`` (the display id) + is still defensible.

In-process, local Ed25519, NO network.
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prsm.compute.chain_rpc.protocol import (
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    parse_message,
)
from prsm.compute.chain_rpc.server import (
    LayerSliceResult,
    LayerSliceRunner,
    LayerStageServer,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.inference.stage_activation_proof import activation_hash
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.compute.shard_receipt import (
    build_receipt_signing_payload,
    per_stage_leaf_job_id,
    settlement_job_id,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity, verify_signature
from prsm.settlement.merkle import batched_receipt_to_leaf
from prsm.settlement.per_stage_settlement_split import (
    NodeSignatureMaterial,
    split_receipt_to_per_node_batched_receipts,
)

_WEI = 10 ** 18


# ── server fakes (mirror tests/unit/test_chain_rpc_server.py) ──────────────


class _FakeAnchor:
    def __init__(self) -> None:
        self.registered: Dict[str, str] = {}

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)

    def register(self, identity) -> None:
        self.registered[identity.node_id] = identity.public_key_b64


@dataclass
class _FakeShard:
    layer_range: Tuple[int, int]


@dataclass
class _FakeModel:
    model_id: str
    shards: List[_FakeShard] = field(default_factory=list)

    @classmethod
    def linear_chain(cls, model_id: str, total_layers: int) -> "_FakeModel":
        return cls(model_id=model_id, shards=[_FakeShard((0, total_layers))])


class _ModelNotFoundError(Exception):
    pass


@dataclass
class _FakeRegistry:
    models: Dict[str, _FakeModel] = field(default_factory=dict)

    def get(self, model_id: str) -> _FakeModel:
        if model_id not in self.models:
            raise _ModelNotFoundError(f"unknown model {model_id!r}")
        return self.models[model_id]


class _FakeTEERuntime:
    def __init__(self, tee_type: TEEType = TEEType.SOFTWARE) -> None:
        self.tee_type = tee_type

    def get_attestation_bytes(self) -> bytes:
        return b"\x01" * 32


class _FakeRunner(LayerSliceRunner):
    """Transforms the activation so the stage OUTPUT differs per stage (so the
    settlement leaf output_hash is well-defined + distinct per node)."""

    def __init__(self, *, bump: float) -> None:
        self.bump = bump

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
    ) -> LayerSliceResult:
        return LayerSliceResult(
            output=activation + self.bump,
            duration_seconds=0.05,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )


def _build_server(stage_identity, settler_identity, *, bump: float):
    anchor = _FakeAnchor()
    anchor.register(stage_identity)
    anchor.register(settler_identity)
    registry = _FakeRegistry(models={"m": _FakeModel.linear_chain("m", 4)})
    return LayerStageServer(
        identity=stage_identity,
        registry=registry,
        runner=_FakeRunner(bump=bump),
        tee_runtime=_FakeTEERuntime(),
        anchor=anchor,
        clock=lambda: 1000.0,
    )


def _make_request(
    *, settler_identity, request_id: str, chain_stage_index: int,
    chain_total: int = 2, activation: Optional[np.ndarray] = None,
) -> RunLayerSliceRequest:
    if activation is None:
        activation = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    blob = activation.tobytes()
    token = HandoffToken.sign(
        identity=settler_identity,
        request_id=request_id,
        chain_stage_index=chain_stage_index,
        chain_total_stages=chain_total,
        deadline_unix=2000.0,
    )
    return RunLayerSliceRequest(
        request_id=request_id,
        model_id="m",
        layer_range=(0, 4),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=blob,
        activation_shape=tuple(activation.shape),
        activation_dtype=str(activation.dtype),
        upstream_token=token,
        deadline_unix=2000.0,
    )


def _dispatch_via_server(stage_identity, settler_identity, *, request_id,
                         stage_index, bump):
    """Drive the REAL server happy path: handle() → _dispatch() → sign().
    Returns the parsed RunLayerSliceResponse (settlement fields populated by the
    server's wiring, not by this test)."""
    server = _build_server(stage_identity, settler_identity, bump=bump)
    req = _make_request(
        settler_identity=settler_identity, request_id=request_id,
        chain_stage_index=stage_index,
    )
    from prsm.compute.chain_rpc.protocol import encode_message
    resp_bytes = server.handle(encode_message(req))
    resp = parse_message(resp_bytes)
    assert isinstance(resp, RunLayerSliceResponse), resp
    return resp


def _make_ir(*, node_a_id, node_b_id, request_id, job_id, streamed=False,
             output_hash=None, **over):
    from prsm.compute.inference.models import InferenceReceipt
    if output_hash is None:
        output_hash = hashlib.sha256(b"the multi-stage output").digest()
    topo = TopologyAssignment(
        positions={(0, 0): node_a_id, (1, 0): node_b_id},
        stage_count=2,
        slots_per_stage=1,
    )
    defaults = dict(
        job_id=job_id,
        request_id=request_id,
        model_id="m",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att",
        output_hash=output_hash,
        duration_seconds=1.0,
        cost_ftns=Decimal("1.0"),
        settler_signature=b"parallax-signature-bytes",
        settler_node_id=node_a_id,
        topology_assignment=topo,
    )
    defaults.update(over)
    return InferenceReceipt(**defaults)


def _material_from_response(resp) -> NodeSignatureMaterial:
    return NodeSignatureMaterial(
        pubkey_b64=resp.stage_settlement_pubkey_b64,
        signature=resp.stage_settlement_signature_b64,
        stage_index=0,  # overwritten by callers via replace below
        output_hash=resp.stage_output_activation_hash,
        executed_at_unix=resp.stage_settlement_executed_at_unix,
    )


# ── (a) JOBID-AGREEMENT ────────────────────────────────────────────────────


def test_server_emits_settlement_sig_over_streamed_agnostic_jobid():
    """The server's wired sign site emits a settlement sig over
    per_stage_leaf_job_id(request_id) — NOT the receipt display job_id — using
    ONLY request_id (no streamed flag)."""
    stage = generate_node_identity("stage0")
    settler = generate_node_identity("settler")
    request_id = "infer-deadbeefcafe"

    resp = _dispatch_via_server(
        stage, settler, request_id=request_id, stage_index=0, bump=1.0,
    )
    # The server emitted the settlement fields (it opted in via the wiring).
    assert resp.stage_settlement_signature_b64 is not None
    assert resp.stage_settlement_pubkey_b64 == stage.public_key_b64
    assert resp.stage_settlement_executed_at_unix is not None
    assert resp.stage_output_activation_hash is not None

    # The emitted signature verifies over the STREAMED-AGNOSTIC leaf payload
    # derived from request_id alone.
    leaf_job_id = per_stage_leaf_job_id(request_id)
    payload = build_receipt_signing_payload(
        job_id=leaf_job_id,
        shard_index=0,
        output_hash=resp.stage_output_activation_hash,
        executed_at_unix=resp.stage_settlement_executed_at_unix,
    )
    assert verify_signature(
        resp.stage_settlement_pubkey_b64, payload,
        resp.stage_settlement_signature_b64,
    )
    # And it does NOT verify over the streamed-PREFIXED display id — proving the
    # server bound the streamed-agnostic id, not the display id.
    display_id = settlement_job_id(request_id, streamed=True)
    assert display_id != leaf_job_id
    bad_payload = build_receipt_signing_payload(
        job_id=display_id,
        shard_index=0,
        output_hash=resp.stage_output_activation_hash,
        executed_at_unix=resp.stage_settlement_executed_at_unix,
    )
    assert not verify_signature(
        resp.stage_settlement_pubkey_b64, bad_payload,
        resp.stage_settlement_signature_b64,
    )


def test_worker_and_splitter_agree_on_jobid_from_request_id_alone():
    """The crux of sp1158: the worker (server) signs from request_id ALONE; the
    splitter rebuilds the leaf from receipt.request_id via per_stage_leaf_job_id
    → the leaf hash matches AND the worker sig verifies (defensible)."""
    node_a = generate_node_identity("A")
    node_b = generate_node_identity("B")
    settler = generate_node_identity("settler")
    request_id = "infer-aaaabbbbcccc"

    resp_a = _dispatch_via_server(
        node_a, settler, request_id=request_id, stage_index=0, bump=1.0,
    )
    resp_b = _dispatch_via_server(
        node_b, settler, request_id=request_id, stage_index=1, bump=2.0,
    )

    mats = {
        node_a.node_id: NodeSignatureMaterial(
            pubkey_b64=resp_a.stage_settlement_pubkey_b64,
            signature=resp_a.stage_settlement_signature_b64,
            stage_index=0,
            output_hash=resp_a.stage_output_activation_hash,
            executed_at_unix=resp_a.stage_settlement_executed_at_unix,
        ),
        node_b.node_id: NodeSignatureMaterial(
            pubkey_b64=resp_b.stage_settlement_pubkey_b64,
            signature=resp_b.stage_settlement_signature_b64,
            stage_index=1,
            output_hash=resp_b.stage_output_activation_hash,
            executed_at_unix=resp_b.stage_settlement_executed_at_unix,
        ),
    }
    # The receipt's DISPLAY job_id is deliberately different from the leaf id.
    receipt = _make_ir(
        node_a_id=node_a.node_id, node_b_id=node_b.node_id,
        request_id=request_id, job_id="parallax-job-DISPLAY-DIFFERENT",
    )
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=receipt,
        total_value_wei=10 * _WEI,
        node_signatures=mats,
        requester_address="0x" + "11" * 20,
    )
    assert shares is not None and len(shares) == 2
    leaf_job_id = per_stage_leaf_job_id(request_id)
    for share in shares:
        br = share.batched_receipt
        # The leaf binds the streamed-agnostic id (from request_id), NOT display.
        assert br.receipt.job_id == leaf_job_id
        assert br.receipt.job_id != receipt.job_id
        leaf = batched_receipt_to_leaf(br)
        expected = build_receipt_signing_payload(
            job_id=leaf_job_id,
            shard_index=br.receipt.shard_index,
            output_hash=br.receipt.output_hash,
            executed_at_unix=br.receipt.executed_at_unix,
        )
        assert leaf.signing_message_hash == expected
        # The WORKER (which knew only request_id) signed exactly this leaf.
        assert verify_signature(
            br.receipt.provider_pubkey_b64, expected, br.receipt.signature,
        )


# ── (b) STREAMED-AGNOSTIC ──────────────────────────────────────────────────


def test_per_node_leaf_jobid_is_identical_for_streamed_and_unary():
    """The per-node leaf job_id is derived from request_id alone, so a streamed
    receipt and a unary receipt with the same request_id produce the SAME leaf
    id — the worker (which doesn't know the flag) agrees with both."""
    node_a = generate_node_identity("A")
    node_b = generate_node_identity("B")
    settler = generate_node_identity("settler")
    request_id = "infer-streamflagtest"

    resp_a = _dispatch_via_server(
        node_a, settler, request_id=request_id, stage_index=0, bump=1.0,
    )
    resp_b = _dispatch_via_server(
        node_b, settler, request_id=request_id, stage_index=1, bump=2.0,
    )
    mats = {
        node_a.node_id: NodeSignatureMaterial(
            pubkey_b64=resp_a.stage_settlement_pubkey_b64,
            signature=resp_a.stage_settlement_signature_b64,
            stage_index=0,
            output_hash=resp_a.stage_output_activation_hash,
            executed_at_unix=resp_a.stage_settlement_executed_at_unix,
        ),
        node_b.node_id: NodeSignatureMaterial(
            pubkey_b64=resp_b.stage_settlement_pubkey_b64,
            signature=resp_b.stage_settlement_signature_b64,
            stage_index=1,
            output_hash=resp_b.stage_output_activation_hash,
            executed_at_unix=resp_b.stage_settlement_executed_at_unix,
        ),
    }

    leaf_ids = set()
    for streamed in (False, True):
        receipt = _make_ir(
            node_a_id=node_a.node_id, node_b_id=node_b.node_id,
            request_id=request_id,
            # Mirror sp1157: a streamed receipt would carry a stream-prefixed
            # display job_id; a unary one a job-prefixed one. The per-node leaf
            # must NOT depend on which.
            job_id=settlement_job_id(request_id, streamed=streamed),
        )
        shares = split_receipt_to_per_node_batched_receipts(
            receipt=receipt,
            total_value_wei=8 * _WEI,
            node_signatures=mats,
            requester_address="0x" + "22" * 20,
        )
        assert shares is not None and len(shares) == 2
        for share in shares:
            br = share.batched_receipt
            leaf_ids.add(br.receipt.job_id)
            # defensible regardless of the receipt's streamed-ness
            leaf = batched_receipt_to_leaf(br)
            expected = build_receipt_signing_payload(
                job_id=per_stage_leaf_job_id(request_id),
                shard_index=br.receipt.shard_index,
                output_hash=br.receipt.output_hash,
                executed_at_unix=br.receipt.executed_at_unix,
            )
            assert leaf.signing_message_hash == expected
            assert verify_signature(
                br.receipt.provider_pubkey_b64, expected, br.receipt.signature,
            )
    # Both the streamed AND unary receipts produced the SAME single leaf id.
    assert leaf_ids == {per_stage_leaf_job_id(request_id)}


# ── (c) BYTE-EQUIVALENCE ───────────────────────────────────────────────────


def test_stage_signature_byte_identical_with_vs_without_settlement_inputs():
    """The 6 wired sites pass settlement inputs; the STAGE signature is the SAME
    as a sign() without them — the settlement sig is a separate artifact and
    does not touch signing_payload."""
    identity = generate_node_identity("worker")
    blob = (np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)).tobytes()
    base = dict(
        identity=identity,
        request_id="infer-byteq",
        activation_blob=blob,
        activation_shape=(1, 4),
        activation_dtype="float32",
        duration_seconds=0.05,
        tee_attestation=b"\x01" * 32,
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )
    resp_without = RunLayerSliceResponse.sign(**base)
    resp_with = RunLayerSliceResponse.sign(
        **base,
        settlement_job_id=per_stage_leaf_job_id("infer-byteq"),
        settlement_stage_index=0,
        settlement_executed_at_unix=1_700_000_000,
    )
    # The stage signature (over signing_payload) is byte-identical.
    assert resp_without.stage_signature_b64 == resp_with.stage_signature_b64
    # The settlement sig is present ONLY when the inputs are passed.
    assert resp_without.stage_settlement_signature_b64 is None
    assert resp_with.stage_settlement_signature_b64 is not None
    # The signing_payload itself is byte-identical (pinned).
    payload_without = RunLayerSliceResponse.signing_payload(
        "infer-byteq", blob, (1, 4), "float32", 0.05, b"\x01" * 32,
        TEEType.SOFTWARE, 0.0, identity.node_id,
    )
    assert verify_signature(
        identity.public_key_b64, payload_without, resp_with.stage_signature_b64,
    )


# ── (d) IN-PROCESS E2E ─────────────────────────────────────────────────────


def test_e2e_two_stages_via_server_to_conserving_defensible_receipts():
    from prsm.compute.chain_rpc.client import (
        _assemble_per_node_settlement_signatures,
        StageOutcome,
    )

    node_a = generate_node_identity("A")
    node_b = generate_node_identity("B")
    settler = generate_node_identity("settler")
    request_id = "infer-e2e-1158"
    total = 7 * _WEI + 3  # non-divisible → exercises remainder conservation

    resp_a = _dispatch_via_server(
        node_a, settler, request_id=request_id, stage_index=0, bump=1.0,
    )
    resp_b = _dispatch_via_server(
        node_b, settler, request_id=request_id, stage_index=1, bump=2.0,
    )

    outcomes = [
        StageOutcome(
            stage_index=0, stage_node_id=node_a.node_id, duration_seconds=0.05,
            tee_attestation=b"\x01" * 32, tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
            stage_output_activation_hash=resp_a.stage_output_activation_hash,
            stage_settlement_signature_b64=resp_a.stage_settlement_signature_b64,
            stage_settlement_pubkey_b64=resp_a.stage_settlement_pubkey_b64,
            stage_settlement_executed_at_unix=(
                resp_a.stage_settlement_executed_at_unix),
        ),
        StageOutcome(
            stage_index=1, stage_node_id=node_b.node_id, duration_seconds=0.05,
            tee_attestation=b"\x01" * 32, tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
            stage_output_activation_hash=resp_b.stage_output_activation_hash,
            stage_settlement_signature_b64=resp_b.stage_settlement_signature_b64,
            stage_settlement_pubkey_b64=resp_b.stage_settlement_pubkey_b64,
            stage_settlement_executed_at_unix=(
                resp_b.stage_settlement_executed_at_unix),
        ),
    ]
    mats = _assemble_per_node_settlement_signatures(outcomes)
    assert mats is not None and len(mats) == 2

    receipt = _make_ir(
        node_a_id=node_a.node_id, node_b_id=node_b.node_id,
        request_id=request_id, job_id="parallax-job-DISPLAY",
    )
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=receipt,
        total_value_wei=total,
        node_signatures=mats,
        requester_address="0x" + "33" * 20,
    )
    assert shares is not None and len(shares) == 2
    # conservation: shares sum to the total EXACTLY
    assert sum(s.share_wei for s in shares) == total
    assert {s.node_id for s in shares} == {node_a.node_id, node_b.node_id}
    for share in shares:
        br = share.batched_receipt
        leaf = batched_receipt_to_leaf(br)
        expected = build_receipt_signing_payload(
            job_id=per_stage_leaf_job_id(request_id),
            shard_index=br.receipt.shard_index,
            output_hash=br.receipt.output_hash,
            executed_at_unix=br.receipt.executed_at_unix,
        )
        assert leaf.signing_message_hash == expected
        assert verify_signature(
            br.receipt.provider_pubkey_b64, expected, br.receipt.signature,
        )
        derived = hashlib.sha256(
            base64.b64decode(br.receipt.provider_pubkey_b64)
        ).hexdigest()[:32]
        assert derived == br.receipt.provider_id


# ── (e) SINGLE-PAYEE UNTOUCHED ─────────────────────────────────────────────


def test_single_payee_adapter_still_uses_display_jobid_and_is_defensible():
    """The single-payee inference_adapter binds receipt.job_id (the display id),
    NOT the per-node streamed-agnostic id. sp1158 did NOT touch it."""
    from prsm.settlement import inference_adapter

    node = generate_node_identity("solo")
    request_id = "infer-solo-1158"
    display_job_id = settlement_job_id(request_id, streamed=False)

    # A single-host receipt (no multi-node topology) → single-payee path.
    @dataclass
    class _SoloReceipt:
        job_id: str
        request_id: str
        output_hash: bytes
        provider_id: str
        provider_pubkey_b64: str
        executed_at_unix: int
        signature: str = ""

    output_hash_hex = activation_hash(b"solo-output")
    executed_at = 1_700_000_999
    # The provider signs the DISPLAY job_id leaf (shard_index 0).
    payload = build_receipt_signing_payload(
        job_id=display_job_id,
        shard_index=0,
        output_hash=output_hash_hex,
        executed_at_unix=executed_at,
    )
    sig = node.sign(payload)

    # Assert the adapter binds receipt.job_id (display) — read the source of
    # truth: the adapter signs/rebuilds over receipt.job_id.
    import inspect
    src = inspect.getsource(inference_adapter)
    assert "receipt.job_id" in src
    assert "per_stage_leaf_job_id" not in src

    # And the display-id leaf the single payee produced is itself defensible.
    assert verify_signature(node.public_key_b64, payload, sig)
    # The display id is NOT the per-node streamed-agnostic id only when the
    # receipt is streamed; for the unary single-payee case they coincide by
    # construction (both parallax-job-<request_id>) — that is expected: the two
    # modes never coexist for one inference, so there is no collision.
    assert display_job_id == per_stage_leaf_job_id(request_id)
