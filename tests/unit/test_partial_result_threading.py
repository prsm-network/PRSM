"""Threading tests for source_agent_pubkey + privacy_budget_consumed
through the PartialResult → SwarmDispatcherAdapter → AggregatorClientAdapter →
SignedPartial chain.

Closes the docstring §1 placeholder follow-on for
`prsm/compute/query_orchestrator/aggregator_client_adapter.py` —
verifies the real values flow through end-to-end without ever hitting
the old hardcoded `b"\\x00" * 32` / `0.0` placeholders.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from prsm.compute.agents.instruction_set import (
    AgentInstruction,
    AgentOp,
    InstructionManifest,
)
from prsm.compute.agents.dispatcher import AgentDispatcher
from prsm.compute.agents.models import DispatchRecord
from prsm.compute.query_orchestrator import (
    AggregatorClientAdapter,
    PartialResult,
    ShardCandidate,
    StakedNode,
    SwarmDispatcherAdapter,
)


# ──────────────────────────────────────────────────────────────────────
# PartialResult schema — defaults + custom values
# ──────────────────────────────────────────────────────────────────────


class TestPartialResultSchema:
    def test_default_source_agent_pubkey_is_32_zero_bytes(self):
        # Backwards-compatible default — older callsites still work.
        p = PartialResult(
            shard_cid="x",
            payload=b"y",
            agent_signature=b"\x00" * 64,
            creator_id="c",
            dp_noise_applied=True,
        )
        assert p.source_agent_pubkey == b"\x00" * 32

    def test_default_privacy_budget_consumed_is_zero(self):
        p = PartialResult(
            shard_cid="x",
            payload=b"y",
            agent_signature=b"\x00" * 64,
            creator_id="c",
            dp_noise_applied=True,
        )
        assert p.privacy_budget_consumed == 0.0

    def test_default_pcu_consumed_is_zero(self):
        # Sp1178 — default 0.0 (→ uniform settlement fallback) when
        # the agent doesn't report PCU.
        p = PartialResult(
            shard_cid="x",
            payload=b"y",
            agent_signature=b"\x00" * 64,
            creator_id="c",
            dp_noise_applied=True,
        )
        assert p.pcu_consumed == 0.0

    def test_custom_source_agent_pubkey_threaded(self):
        real_pubkey = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes_raw()
        p = PartialResult(
            shard_cid="x",
            payload=b"y",
            agent_signature=b"\x00" * 64,
            creator_id="c",
            dp_noise_applied=True,
            source_agent_pubkey=real_pubkey,
            privacy_budget_consumed=0.42,
            pcu_consumed=12.5,
        )
        assert p.source_agent_pubkey == real_pubkey
        assert p.privacy_budget_consumed == 0.42
        assert p.pcu_consumed == 12.5


# ──────────────────────────────────────────────────────────────────────
# SwarmDispatcherAdapter — threads result-dict fields → PartialResult
# ──────────────────────────────────────────────────────────────────────


class TestSwarmDispatcherAdapterThreading:
    @pytest.mark.asyncio
    async def test_dispatcher_result_dict_carries_pubkey_and_budget_into_partial(self):
        real_pubkey = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes_raw()
        result_dict = {
            "payload": b"agent-output",
            "agent_signature": b"\x11" * 64,
            "dp_noise_applied": True,
            "source_agent_pubkey": real_pubkey,
            "privacy_budget_consumed": 0.15,
            # Sp1178 — the agent's measured PCU (AgentExecutor emits
            # it under the "pcu" key). Pre-fix the adapter dropped it.
            "pcu": 3.5,
        }

        # Mock the AgentDispatcher
        agent_dispatcher = MagicMock(spec=AgentDispatcher)
        agent_dispatcher.dispatch = AsyncMock(
            return_value=DispatchRecord(
                agent_id="agent-1",
                origin_node="origin-x",
                target_node="",
                ftns_budget=100.0,
                status="bidding",
            )
        )
        agent_dispatcher.wait_for_result = AsyncMock(return_value=result_dict)

        adapter = SwarmDispatcherAdapter(
            agent_dispatcher=agent_dispatcher,
            per_shard_budget_ftns=100,
            wasm_executor_binary=b"\x00asm\x01\x00\x00\x00",  # WASM magic + version
        )

        manifest = InstructionManifest(
            query="x",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        shards = [
            ShardCandidate(
                cid="prsm:s-0",
                similarity=0.9,
                creator_id="creator-A",
                holder_node_ids=("node-1",),
            ),
        ]

        partials = await adapter.fan_out(manifest, shards)

        assert len(partials) == 1
        p = partials[0]
        assert p.source_agent_pubkey == real_pubkey
        assert p.privacy_budget_consumed == 0.15
        # Sp1178 — PCU now threads from the result dict into the
        # PartialResult (was silently dropped → always 0.0 → settlement
        # always uniform despite agents reporting real PCU).
        assert p.pcu_consumed == 3.5

    @pytest.mark.asyncio
    async def test_dispatcher_omits_pubkey_falls_back_to_default(self):
        # Older dispatchers may not surface the new fields; defaults
        # (zero bytes / 0.0) preserve backwards compatibility.
        result_dict = {
            "payload": b"agent-output",
            "agent_signature": b"\x11" * 64,
            "dp_noise_applied": True,
            # No source_agent_pubkey, no privacy_budget_consumed
        }
        agent_dispatcher = MagicMock(spec=AgentDispatcher)
        agent_dispatcher.dispatch = AsyncMock(
            return_value=DispatchRecord(
                agent_id="agent-2",
                origin_node="origin-x",
                target_node="",
                ftns_budget=100.0,
                status="bidding",
            )
        )
        agent_dispatcher.wait_for_result = AsyncMock(return_value=result_dict)

        adapter = SwarmDispatcherAdapter(
            agent_dispatcher=agent_dispatcher,
            per_shard_budget_ftns=100,
            wasm_executor_binary=b"\x00asm\x01\x00\x00\x00",  # WASM magic + version
        )
        partials = await adapter.fan_out(
            InstructionManifest(
                query="x",
                instructions=[AgentInstruction(op=AgentOp.COUNT)],
            ),
            [ShardCandidate(
                cid="prsm:s-0",
                similarity=0.9,
                creator_id="c",
                holder_node_ids=("n",),
            )],
        )

        assert len(partials) == 1
        assert partials[0].source_agent_pubkey == b"\x00" * 32
        assert partials[0].privacy_budget_consumed == 0.0
        # Sp1178 — no "pcu" key reported → defaults to 0.0 (→ uniform
        # settlement fallback), preserving backwards compatibility.
        assert partials[0].pcu_consumed == 0.0


# ──────────────────────────────────────────────────────────────────────
# AggregatorClientAdapter — threads PartialResult fields → SignedPartial
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _CapturingTransport:
    """Captures the AggregateRequest the adapter constructs so we can
    inspect the SignedPartials it built. Uses a fixed aggregator
    keypair so the response's commit verifies against the
    aggregator_pubkey_hash the adapter expects."""

    aggregator_privkey: ed25519.Ed25519PrivateKey
    aggregator_pubkey_hash: bytes
    captured_request: object = None

    async def send(self, aggregator_node_id, request, timeout_seconds):
        from prsm.compute.query_orchestrator import (
            AggregateResponse,
            AggregationCommit,
        )
        self.captured_request = request

        plaintext = b"ok"
        digest = hashlib.sha256(plaintext).digest()
        commit = AggregationCommit(
            query_id=request.query_id,
            # MUST match the aggregator the adapter was called with —
            # otherwise the adapter raises INVALID_AGGREGATOR_IDENTITY.
            aggregator_pubkey_hash=self.aggregator_pubkey_hash,
            result_digest=digest,
        )
        commit_sig = self.aggregator_privkey.sign(commit.signing_payload())
        return AggregateResponse(
            request_id=request.request_id,
            query_id=request.query_id,
            commit=commit,
            commit_signature=commit_sig,
            encrypted_plaintext=plaintext,
            nonce=b"\x00" * 24,
            aggregator_pubkey=self.aggregator_privkey.public_key().public_bytes_raw(),
            privacy_budget_consumed=0.0,
            contributing_creators=("c",),
            completed_unix=1,
        )


class TestAggregatorClientThreading:
    """The adapter's `aggregate(...)` call MUST construct SignedPartials
    using the PartialResult's source_agent_pubkey + privacy_budget_consumed
    fields directly — no more hardcoded placeholders."""

    @pytest.mark.asyncio
    async def test_signed_partial_carries_real_pubkey_and_budget(self):
        # Build a PartialResult with non-default fields.
        real_pubkey = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes_raw()
        partial = PartialResult(
            shard_cid="prsm:shard-X",
            payload=b"the-output",
            agent_signature=b"\x22" * 64,
            creator_id="creator-Z",
            dp_noise_applied=True,
            source_agent_pubkey=real_pubkey,
            privacy_budget_consumed=0.27,
        )

        # Build the aggregator + matching transport.
        aggregator_priv = ed25519.Ed25519PrivateKey.generate()
        aggregator_pubkey = aggregator_priv.public_key().public_bytes_raw()
        aggregator_pubkey_hash = hashlib.sha256(aggregator_pubkey).digest()
        aggregator = StakedNode(
            node_id="agg-1",
            pubkey_hash=aggregator_pubkey_hash,
            stake_amount_ftns=1000,
            tier="T2",
            has_tee=False,
            reputation_score=1.0,
        )
        transport = _CapturingTransport(
            aggregator_privkey=aggregator_priv,
            aggregator_pubkey_hash=aggregator_pubkey_hash,
        )
        prompter_priv = ed25519.Ed25519PrivateKey.generate()
        adapter = AggregatorClientAdapter(
            prompter_pubkey=prompter_priv.public_key().public_bytes_raw(),
            prompter_node_id="prompter-1",
            prompter_signer=prompter_priv.sign,
            beacon_provider=lambda: b"\x33" * 32,
            transport=transport,
        )
        manifest = InstructionManifest(
            query="anything",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )

        await adapter.aggregate(
            aggregator=aggregator,
            manifest=manifest,
            partials=[partial],
            query_id=b"q" * 32,
        )

        # The SignedPartial in the captured request MUST carry the
        # real values, not the old placeholders.
        assert transport.captured_request is not None
        signed = transport.captured_request.partials[0]
        assert signed.source_agent_pubkey == real_pubkey
        assert signed.privacy_budget_consumed == pytest.approx(0.27)

    @pytest.mark.asyncio
    async def test_default_partial_still_works(self):
        # Backwards compat: a PartialResult constructed without the
        # new fields defaults to zero bytes / 0.0, and the adapter
        # threads those defaults through unchanged.
        partial = PartialResult(
            shard_cid="prsm:shard-X",
            payload=b"the-output",
            agent_signature=b"\x22" * 64,
            creator_id="creator-Z",
            dp_noise_applied=True,
        )
        aggregator_priv = ed25519.Ed25519PrivateKey.generate()
        aggregator_pubkey = aggregator_priv.public_key().public_bytes_raw()
        aggregator_pubkey_hash = hashlib.sha256(aggregator_pubkey).digest()
        aggregator = StakedNode(
            node_id="agg-1",
            pubkey_hash=aggregator_pubkey_hash,
            stake_amount_ftns=1000,
            tier="T2",
            has_tee=False,
            reputation_score=1.0,
        )
        transport = _CapturingTransport(
            aggregator_privkey=aggregator_priv,
            aggregator_pubkey_hash=aggregator_pubkey_hash,
        )
        prompter_priv = ed25519.Ed25519PrivateKey.generate()
        adapter = AggregatorClientAdapter(
            prompter_pubkey=prompter_priv.public_key().public_bytes_raw(),
            prompter_node_id="prompter-1",
            prompter_signer=prompter_priv.sign,
            beacon_provider=lambda: b"\x33" * 32,
            transport=transport,
        )
        manifest = InstructionManifest(
            query="x",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        await adapter.aggregate(
            aggregator=aggregator,
            manifest=manifest,
            partials=[partial],
            query_id=b"q" * 32,
        )
        signed = transport.captured_request.partials[0]
        assert signed.source_agent_pubkey == b"\x00" * 32
        assert signed.privacy_budget_consumed == 0.0
