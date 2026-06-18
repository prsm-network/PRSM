"""Sprint 1153 — automated cross-host inference→settlement loop, end-to-end.

Tier-1's headline ("prove the network across real hosts") was PROVEN LIVE on
2026-06-06: a 2-stage gpt2 split across two Lambda A10s, per-stage Ed25519
receipts verifying against the Base anchor, on-chain settlement on mainnet. But
that proof was a MANUAL one-off. The individual bricks (per-stage receipt
signing, activation-chain verify, the inference→settlement adapter, the
accumulator, the BatchSettlementClient commit/finalize, the sp1031 per-stage
settlement split) are each unit-tested IN ISOLATION — there was NO automated
test exercising the FULL assembled loop with DISTINCT per-stage signing nodes
all the way through settlement. A regression in how those bricks COMPOSE would
not be caught.

This module is the repeatable form of that manual bench: one deterministic,
in-process integration test that assembles the whole loop and asserts every
hand-off, reusing the REAL production primitives and faking ONLY (a) the model
compute (we hash fixed activation blobs instead of running gpt2) and (b) the
chain RPC (an AsyncMock SettlementContractClient returning canned commit/status,
the same fake pattern the existing settlement tests use).

THE LOOP (each step = real production code):
  1. Two DISTINCT NodeIdentity signers — nodeA (stage 0), nodeB (stage 1). This
     is the cross-host essence: different keypairs per stage.
  2. A 2-stage signed §7 InferenceReceipt carrying a StageActivationChain signed
     stage0→nodeA, stage1→nodeB, with continuity (stage0.output == stage1.input)
     and a topology_assignment naming stage0→nodeA / stage1→nodeB. Built via the
     REAL sign path (sign_receipt + StageActivationProof.signing_bytes).
  3. F1/F2 VERIFY — verify_stage_activation_chain (continuity + per-stage sigs +
     anti-splice request_id binding), chain_supports_topology (F2), and the
     composed off-chain verdict verify_inference_receipt_for_challenge.
  4. ADAPT — inference_receipt_to_batched_receipt re-signs the shard payload into
     a challenge-defensible BatchedReceipt.
  5. ACCUMULATE → COMMIT → FINALIZE — the BatchedReceipt flows through the real
     ReceiptAccumulator + BatchSettlementClient against a FAKE chain client.
  6. PER-STAGE SPLIT (sp1031) — settle_inference_receipt fans the off-chain
     release across BOTH topology node_ids via the real PaymentEscrow +
     LocalLedger; we assert nodeA AND nodeB are each actually credited.

NON-VACUITY (these negatives are proven to FLIP the result, below):
  (a) stage1 signed by the WRONG key → per-stage-sig verify FAILS.
  (b) broken chain (stage0.output != stage1.input) → continuity FAILS.
  (c) topology that lies about who ran a stage → F2 cross-check FAILS.
  (d) a head-only settlement split → the both-nodes-credited assertion FAILS.

Determinism: fixed keys (generated in-process, no persistence), fixed inputs,
LocalLedger on an in-memory SQLite db, an AsyncMock chain client with canned
returns. No network/ports/sleep/wall-clock-race/GPU. Runs in well under a second.
"""
from __future__ import annotations

import dataclasses
import hashlib
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from prsm.node.identity import generate_node_identity
from prsm.compute.inference.models import InferenceReceipt, ContentTier
from prsm.compute.inference.receipt import sign_receipt, verify_receipt
from prsm.compute.inference.stage_activation_proof import (
    StageActivationChain,
    StageActivationProof,
    activation_hash,
    chain_supports_topology,
    verify_stage_activation_chain,
)
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.settlement.challenge_verifier import (
    ChallengeReason,
    verify_inference_receipt_for_challenge,
)
from prsm.settlement.inference_adapter import inference_receipt_to_batched_receipt
from prsm.settlement.accumulator import (
    AccumulatorConfig,
    ReceiptAccumulator,
    TriggerReason,
)
from prsm.settlement.client import BatchSettlementClient


# ── fixed, deterministic inputs (no wall-clock, no randomness in the asserts) ──
REQUEST_ID = "infer-crosshost-1153"
JOB_ID = "job-crosshost-1153"
ESCROW_ID = "esc-crosshost-1153"
EXECUTED_AT = 1_700_000_000
ONE_FTNS_WEI = 10 ** 18
COST = Decimal("1.0")

REQUESTER_ADDR = "0x" + "a" * 40
PROVIDER_ADDR = "0x" + "b" * 40

# The activation blobs the live bench would have hashed (model compute = FAKE):
#   A0 = prompt encoded to activations (stage 0 input)
#   A1 = stage 0 output == stage 1 input (the cross-host hand-off activation)
#   A2 = stage 1 output (decoded to the response)
A0 = activation_hash(b"prompt-activations")
A1 = activation_hash(b"midpoint-activations")
A2 = activation_hash(b"final-activations")

# The receipt's output_hash is a raw 32-byte sha256 (adapter requires raw bytes).
OUTPUT_HASH_RAW = hashlib.sha256(b"final-activations").digest()
PROMPT_HASH_RAW = hashlib.sha256(b"the-prompt").digest()


def _make_two_distinct_identities():
    """nodeA runs stage 0, nodeB runs stage 1 — DISTINCT Ed25519 keypairs.
    This is the cross-host signer separation the whole loop is about."""
    node_a = generate_node_identity("crosshost-nodeA")
    node_b = generate_node_identity("crosshost-nodeB")
    assert node_a.node_id != node_b.node_id
    return node_a, node_b


def _signed_stage_proof(identity, *, request_id, stage_index, h_in, h_out):
    """Sign a StageActivationProof with the REAL worker sign path."""
    unsigned = StageActivationProof(
        stage_index=stage_index,
        stage_node_id=identity.node_id,
        input_activation_hash=h_in,
        output_activation_hash=h_out,
    )
    sig = identity.sign(unsigned.signing_bytes(request_id))
    return StageActivationProof(
        stage_index=stage_index,
        stage_node_id=identity.node_id,
        input_activation_hash=h_in,
        output_activation_hash=h_out,
        stage_signature_b64=sig,
    )


def _build_chain(node_a, node_b, *, request_id=REQUEST_ID, break_continuity=False):
    """A 2-stage activation chain: stage0→nodeA (A0→A1), stage1→nodeB (A1→A2).
    With break_continuity=True, stage1's input is a DIFFERENT hash than stage0's
    output (the negative-control broken-chain case)."""
    stage1_in = activation_hash(b"spliced-mismatch") if break_continuity else A1
    return StageActivationChain(
        request_id=request_id,
        proofs=[
            _signed_stage_proof(
                node_a, request_id=request_id, stage_index=0, h_in=A0, h_out=A1),
            _signed_stage_proof(
                node_b, request_id=request_id, stage_index=1, h_in=stage1_in, h_out=A2),
        ],
    )


def _topology(node_a_id, node_b_id):
    """topology_assignment naming stage0→nodeA, stage1→nodeB (1 slot each)."""
    return TopologyAssignment(
        positions={(0, 0): node_a_id, (1, 0): node_b_id},
        stage_count=2,
        slots_per_stage=1,
    )


def _build_signed_receipt(node_a, node_b, *, chain=None, topology=None, settler=None):
    """Assemble + sign the §7 InferenceReceipt for the 2-stage cross-host run.

    The SETTLER signs the whole receipt (binding the activation-chain stable_hash
    + the topology into signing_payload). By default the settler is nodeA — the
    head/orchestrator node, exactly as the live bench (the head builds + signs the
    receipt, the per-stage proofs carry the distinct worker signatures)."""
    settler = settler or node_a
    chain = chain if chain is not None else _build_chain(node_a, node_b)
    topology = topology if topology is not None else _topology(node_a.node_id, node_b.node_id)
    unsigned = InferenceReceipt(
        job_id=JOB_ID,
        request_id=REQUEST_ID,
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att",
        output_hash=OUTPUT_HASH_RAW,
        duration_seconds=2.5,
        cost_ftns=COST,
        prompt_hash=PROMPT_HASH_RAW,
        topology_assignment=topology,
        stage_activation_chain=chain,
    )
    return sign_receipt(unsigned, settler)


def _anchor_keys(node_a, node_b):
    """The on-chain anchor's {node_id: pubkey_b64} the verifier resolves against."""
    return {
        node_a.node_id: node_a.public_key_b64,
        node_b.node_id: node_b.public_key_b64,
    }


# ── real PaymentEscrow + LocalLedger for the off-chain per-stage split (sp1031) ─


async def _fund_and_escrow(node_a, node_b):
    """Build a REAL PaymentEscrow over a REAL in-memory LocalLedger, fund the
    requester, and lock the job's escrow. Returns (escrow, ledger, requester_id)."""
    from prsm.node.local_ledger import LocalLedger, TransactionType
    from prsm.node.payment_escrow import PaymentEscrow

    ledger = LocalLedger(db_path=":memory:")
    await ledger.initialize()
    requester_id = "requester-wallet"
    # Fund the requester so create_escrow can lock the budget (real ledger credit).
    await ledger.credit(
        requester_id, float(COST), TransactionType.REWARD, description="test-fund")
    escrow = PaymentEscrow(ledger=ledger, node_id=node_a.node_id)
    locked = await escrow.create_escrow(JOB_ID, float(COST), requester_id=requester_id)
    assert locked is not None, "escrow must lock the funded budget"
    return escrow, ledger, requester_id


def _make_fake_chain_client():
    """An AsyncMock SettlementContractClient (the FAKE chain RPC) returning canned
    commit/status — mirrors test_batch_settlement_client._make_mock_contract."""
    contract = AsyncMock()
    batch_id = hashlib.sha256(b"crosshost-1153-batch").digest()

    async def _commit(**kwargs):
        # Echo enough to assert the merkle root + value flow through.
        return (batch_id, EXECUTED_AT)

    contract.commit_batch.side_effect = _commit
    contract.is_finalizable.return_value = True
    contract.finalize_batch.return_value = None
    # PENDING on the pre-check, FINALIZED after finalize.
    contract.get_batch_status.return_value = 2
    return contract, batch_id


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# ════════════════════════════════════════════════════════════════════════════
# THE FULL LOOP — assembled end to end
# ════════════════════════════════════════════════════════════════════════════


def test_full_crosshost_loop_two_nodes_through_settlement():
    """The headline test: 2 distinct signers → signed 2-stage receipt + chain →
    F1/F2 verify → adapt → accumulate → commit → finalize → BOTH nodes credited.
    Every step uses real production code; only model compute + chain RPC faked."""
    node_a, node_b = _make_two_distinct_identities()

    # ── Step 2: the signed 2-stage receipt + activation chain ──────────────────
    receipt = _build_signed_receipt(node_a, node_b)
    # The settler (nodeA) signature verifies over the whole receipt.
    assert verify_receipt(receipt, public_key_b64=node_a.public_key_b64)
    # Distinct per-stage signers are recorded in the signed chain.
    chain = receipt.stage_activation_chain
    assert chain.topology_node_ids() == [node_a.node_id, node_b.node_id]
    assert node_a.node_id != node_b.node_id

    # ── Step 3: F1/F2 verify (continuity + per-stage sigs + anti-splice + F2) ──
    anchor_keys = _anchor_keys(node_a, node_b)

    class _Anchor:
        def lookup(self, node_id):
            return anchor_keys.get(node_id)

    ok, why = verify_stage_activation_chain(
        chain,
        prompt_hash_hex=A0,
        output_hash_hex=A2,
        anchor=_Anchor(),
        expected_request_id=receipt.request_id,   # sp1112 anti-splice binding
    )
    assert ok, f"F1 chain verify failed: {why}"

    topo_ok, topo_why = chain_supports_topology(chain, receipt.topology_assignment)
    assert topo_ok, f"F2 topology cross-check failed: {topo_why}"

    # Composed off-chain verdict (challenge_verifier) — receipt_ok with no proven
    # fraud ground, across settler-sig + chain + topology + request_id binding.
    report = verify_inference_receipt_for_challenge(
        receipt,
        settler_public_key_b64=node_a.public_key_b64,
        stage_public_keys=anchor_keys,
    )
    assert report.receipt_ok, f"challenge verdict flagged fraud: {report.to_dict()}"
    assert report.proven_findings() == []

    # ── Step 4: ADAPT — InferenceReceipt → challenge-defensible BatchedReceipt ──
    # The adapter requires identity == receipt.settler (nodeA built + signed it).
    batched = inference_receipt_to_batched_receipt(
        receipt=receipt,
        identity=node_a,
        requester_address=REQUESTER_ADDR,
        provider_address=PROVIDER_ADDR,
        value_ftns=ONE_FTNS_WEI,
        local_escrow_id=ESCROW_ID,
        executed_at_unix=EXECUTED_AT,
    )
    # The re-signed shard leaf is reconstructable + verifies (challenge-defensible).
    from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
    from prsm.compute.shard_receipt import build_receipt_signing_payload
    from prsm.node.identity import verify_signature
    leaf_hash = hash_leaf(batched_receipt_to_leaf(batched))
    assert len(leaf_hash) == 32
    payload = build_receipt_signing_payload(
        job_id=JOB_ID, shard_index=0,
        output_hash=OUTPUT_HASH_RAW.hex(), executed_at_unix=EXECUTED_AT)
    assert verify_signature(
        batched.receipt.provider_pubkey_b64, payload, batched.receipt.signature)
    assert batched.value_ftns == ONE_FTNS_WEI

    # ── Step 5: ACCUMULATE → COMMIT → FINALIZE against the FAKE chain client ───
    contract, expected_batch_id = _make_fake_chain_client()
    cfg = AccumulatorConfig(
        count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10 ** 30)
    acc = ReceiptAccumulator(cfg)
    # The committer is bound to the provider address (msg.sender of commitBatch).
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)

    _run(client.accumulate(batched))
    assert acc.total_receipt_count() == 1

    committed = _run(client.commit_ready_batches())
    assert len(committed) == 1
    cb = committed[0]
    assert cb.batch_id == expected_batch_id
    assert cb.receipt_count == 1
    assert cb.total_value_ftns == ONE_FTNS_WEI
    assert cb.trigger_reason == TriggerReason.COUNT
    # The committed merkle root matches the independently recomputed root.
    from prsm.settlement.merkle import build_tree_and_proofs
    expected_root = build_tree_and_proofs([leaf_hash]).root
    commit_kwargs = contract.commit_batch.await_args.kwargs
    assert commit_kwargs["merkle_root"] == expected_root
    assert acc.total_receipt_count() == 0   # accumulator drained on success

    finalized = _run(client.finalize_ready_batches())
    assert len(finalized) == 1
    assert finalized[0].tx_submitted is True
    contract.finalize_batch.assert_awaited_once()
    assert client.is_finalized_locally(cb.batch_id)

    # ── Step 6: PER-STAGE SPLIT — BOTH nodeA AND nodeB credited off-chain ──────
    from prsm.economy.credit_policy import settle_inference_receipt
    escrow, ledger, requester_id = _run(_fund_and_escrow(node_a, node_b))
    decision = _run(settle_inference_receipt(
        payment_escrow=escrow, receipt=receipt,
        operator_id=node_a.node_id, job_id=JOB_ID,
    ))
    assert decision.release_to_operator == COST
    # The split must credit BOTH stage nodes — not just the settler/head.
    bal_a = _run(ledger.get_balance(node_a.node_id))
    bal_b = _run(ledger.get_balance(node_b.node_id))
    assert bal_a == pytest.approx(0.5), f"nodeA (head) credit wrong: {bal_a}"
    assert bal_b == pytest.approx(0.5), f"nodeB (worker) credit wrong: {bal_b}"
    # The bug this guards: the worker (nodeB) MUST be paid, not zero.
    assert bal_b > 0, "cross-host worker nodeB was not credited (sp1031 regression)"
    # Conservation: the two credits sum to exactly the released amount.
    assert bal_a + bal_b == pytest.approx(float(COST))


# ════════════════════════════════════════════════════════════════════════════
# NON-VACUITY — each negative is proven to FLIP the loop's asserted invariant
# ════════════════════════════════════════════════════════════════════════════


def test_negative_stage1_signed_by_wrong_key_fails_perstage_sig():
    """(a) sig-tamper: stage 1 signed by a THIRD key the anchor doesn't publish
    for nodeB → the chain that verified in the happy path now FAILS per-stage-sig
    verify, and the challenge verdict proves an INVALID_ACTIVATION_CHAIN ground."""
    node_a, node_b = _make_two_distinct_identities()
    impostor = generate_node_identity("impostor")

    # stage 1 CLAIMS to be nodeB but is signed by the impostor's key.
    bad_stage1 = StageActivationProof(
        stage_index=1, stage_node_id=node_b.node_id,
        input_activation_hash=A1, output_activation_hash=A2,
        stage_signature_b64=impostor.sign(
            StageActivationProof(1, node_b.node_id, A1, A2).signing_bytes(REQUEST_ID)),
    )
    good_stage0 = _signed_stage_proof(
        node_a, request_id=REQUEST_ID, stage_index=0, h_in=A0, h_out=A1)
    tampered_chain = StageActivationChain(REQUEST_ID, [good_stage0, bad_stage1])

    anchor_keys = _anchor_keys(node_a, node_b)

    class _Anchor:
        def lookup(self, node_id):
            return anchor_keys.get(node_id)

    # FLIP: the same verify that passed in the happy path now fails.
    ok, why = verify_stage_activation_chain(
        tampered_chain, prompt_hash_hex=A0, output_hash_hex=A2,
        anchor=_Anchor(), expected_request_id=REQUEST_ID)
    assert not ok
    assert "signature invalid" in why

    # And the composed verdict proves it as an on-receipt fraud ground.
    receipt = _build_signed_receipt(node_a, node_b, chain=tampered_chain)
    report = verify_inference_receipt_for_challenge(
        receipt, settler_public_key_b64=node_a.public_key_b64,
        stage_public_keys=anchor_keys)
    assert not report.receipt_ok
    assert any(
        f.reason == ChallengeReason.INVALID_ACTIVATION_CHAIN and f.proven
        for f in report.findings)


def test_negative_broken_chain_fails_continuity():
    """(b) broken chain: stage0.output_hash != stage1.input_hash → continuity
    FAILS (the spliced-intermediate defense). FLIPS the happy path's F1 assert."""
    node_a, node_b = _make_two_distinct_identities()
    broken = _build_chain(node_a, node_b, break_continuity=True)

    anchor_keys = _anchor_keys(node_a, node_b)

    class _Anchor:
        def lookup(self, node_id):
            return anchor_keys.get(node_id)

    ok, why = verify_stage_activation_chain(
        broken, prompt_hash_hex=A0, output_hash_hex=A2,
        anchor=_Anchor(), expected_request_id=REQUEST_ID)
    assert not ok
    assert "chain broken between stage 0 and 1" in why

    # The off-chain verdict flags INVALID_ACTIVATION_CHAIN (internal links broken).
    receipt = _build_signed_receipt(node_a, node_b, chain=broken)
    report = verify_inference_receipt_for_challenge(
        receipt, settler_public_key_b64=node_a.public_key_b64,
        stage_public_keys=anchor_keys)
    assert not report.receipt_ok
    assert any(
        f.reason == ChallengeReason.INVALID_ACTIVATION_CHAIN and f.proven
        for f in report.findings)


def test_negative_topology_that_lies_fails_f2_crosscheck():
    """(c) topology-lie: the asserted topology_assignment claims stage 1 was run
    by some OTHER node than the one that actually signed → F2 cross-check FAILS.
    FLIPS the happy path's chain_supports_topology assert + the verdict's
    ACTIVATION_TOPOLOGY_MISMATCH ground."""
    node_a, node_b = _make_two_distinct_identities()
    chain = _build_chain(node_a, node_b)
    lying_topology = _topology(node_a.node_id, "someone-else-entirely")

    topo_ok, topo_why = chain_supports_topology(chain, lying_topology)
    assert not topo_ok
    assert "topology does not assign" in topo_why

    receipt = _build_signed_receipt(
        node_a, node_b, chain=chain, topology=lying_topology)
    report = verify_inference_receipt_for_challenge(
        receipt, settler_public_key_b64=node_a.public_key_b64,
        stage_public_keys=_anchor_keys(node_a, node_b))
    assert not report.receipt_ok
    assert any(
        f.reason == ChallengeReason.ACTIVATION_TOPOLOGY_MISMATCH and f.proven
        for f in report.findings)


def test_negative_head_only_split_fails_both_credited_assertion():
    """(d) head-only split: a settlement that credits ONLY the head node (nodeA)
    leaves the worker (nodeB) at zero → the both-nodes-credited invariant FAILS.
    Proven by driving the REAL escrow with the single-payee path and showing nodeB
    gets nothing, then showing the per-stage split path fixes it."""
    node_a, node_b = _make_two_distinct_identities()

    # Head-only: release_escrow credits ONLY nodeA (the single-payee bug shape).
    escrow_head, ledger_head, _ = _run(_fund_and_escrow(node_a, node_b))
    _run(escrow_head.release_escrow(JOB_ID, node_a.node_id))
    bal_a_head = _run(ledger_head.get_balance(node_a.node_id))
    bal_b_head = _run(ledger_head.get_balance(node_b.node_id))
    assert bal_a_head == pytest.approx(float(COST))   # head got everything
    assert bal_b_head == 0.0                           # worker got nothing
    # The happy-path invariant (both credited 0.5/0.5) would FAIL here:
    both_credited = (
        bal_a_head == pytest.approx(0.5) and bal_b_head == pytest.approx(0.5))
    assert not both_credited, "head-only split must NOT satisfy both-credited"

    # Now the real per-stage split path (sp1031) genuinely credits BOTH.
    from prsm.economy.credit_policy import settle_inference_receipt
    receipt = _build_signed_receipt(node_a, node_b)
    escrow_split, ledger_split, _ = _run(_fund_and_escrow(node_a, node_b))
    _run(settle_inference_receipt(
        payment_escrow=escrow_split, receipt=receipt,
        operator_id=node_a.node_id, job_id=JOB_ID))
    bal_a = _run(ledger_split.get_balance(node_a.node_id))
    bal_b = _run(ledger_split.get_balance(node_b.node_id))
    assert bal_a == pytest.approx(0.5)
    assert bal_b == pytest.approx(0.5)   # the worker is now paid — the fix holds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
