"""Phase 3.x.11.y Task 5 — RpcChainExecutor speculative-decoding
tests.

Covers ``draft_model=`` opt-in path on ``execute_chain_streaming``:
  - Construction validation (greedy-only, depth bounds, draft
    requires sharded_decode, callable rollback transport)
  - Multi-tokens-per-iteration on full accept (K+1 emitted per
    VERIFY round)
  - All-rejected fallback (only the verifier's correction emits)
  - Partial accept (longest matching prefix)
  - max_tokens cap honored across speculation rounds (including
    mid-emit truncation)
  - EOS in emitted tokens terminates the loop
  - Cancellation propagates draft.evict + cache eviction broadcast
    even when caller closes the generator early
  - Greedy-equivalence: speculation output is bit-identical to
    Phase 3.x.11 single-token decode for the same prompt + temp=0
  - Rollback broadcast triggers on partial accept; not triggered
    when accepted_count == K
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc.client import (
    ChainExecutionError,
    ExecutorErrorCode,
    RpcChainExecutor,
    StreamToken,
)
from prsm.compute.chain_rpc.protocol import (
    DecodeMode,
    EvictCacheRequest,
    EvictCacheResponse,
    RollbackCacheRequest,
    RollbackCacheResponse,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    encode_message,
    parse_message,
    response_input_commitment_for_request,
)
from prsm.compute.chain_rpc.activation_codec import (
    decode_activation,
    encode_activation,
)
from prsm.compute.inference.models import ContentTier, InferenceRequest
from prsm.compute.inference.parallax_executor import ChainExecutionResult
from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class _FakeAnchor:
    def __init__(self) -> None:
        self.registered: Dict[str, str] = {}

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


class _FakeTokenizer:
    def encode(self, text: str) -> List[int]:
        words = text.split()
        return [max(1, len(w)) for w in words] or [1]

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return " ".join(f"tok{tid}" for tid in ids)


class _SpecStageSim:
    """Per-stage VERIFY-aware sim. PREFILL/INCREMENTAL behave like
    the Phase 3.x.11 sim (echo input + tail samples script).
    VERIFY: tail returns scripted (verified_token_ids,
    accepted_count, is_terminal) per round; non-tail just echoes
    K+1 hidden states.

    Scripts:
      - sample_script:   List[(token, is_terminal)] for PREFILL +
                         INCREMENTAL tail responses.
      - verify_script:   List[(verified_tuple, accepted_count,
                         is_terminal)] for VERIFY tail responses.
    """

    def __init__(
        self,
        identity,
        *,
        is_tail: bool,
        sample_script: Optional[List[Tuple[int, bool]]] = None,
        verify_script: Optional[
            List[Tuple[Tuple[int, ...], int, bool]]
        ] = None,
    ) -> None:
        self.identity = identity
        self.is_tail = is_tail
        self._sample_script = list(sample_script or [])
        self._sample_cursor = 0
        self._verify_script = list(verify_script or [])
        self._verify_cursor = 0
        self.requests: List[RunLayerSliceRequest] = []

    def handle(self, request_bytes: bytes) -> bytes:
        request = parse_message(request_bytes)
        assert isinstance(request, RunLayerSliceRequest)
        self.requests.append(request)

        in_arr = decode_activation(
            request.activation_blob,
            request.activation_shape,
            request.activation_dtype,
        )
        out_arr = in_arr.astype(np.float32)
        out_blob, out_shape, out_dtype = encode_activation(out_arr)

        next_token_id: Optional[int] = None
        is_terminal = False
        verified_token_ids: Optional[Tuple[int, ...]] = None
        accepted_count: Optional[int] = None

        if self.is_tail:
            if request.decode_mode == DecodeMode.VERIFY:
                if self._verify_cursor >= len(self._verify_script):
                    raise AssertionError(
                        "_SpecStageSim: verify_script exhausted "
                        "(test bug)"
                    )
                v, ac, term = self._verify_script[self._verify_cursor]
                self._verify_cursor += 1
                verified_token_ids = tuple(int(t) for t in v)
                accepted_count = int(ac)
                is_terminal = bool(term)
                # next_token_id = last emitted (= verified[ac]).
                next_token_id = int(verified_token_ids[accepted_count])
            else:
                if self._sample_cursor >= len(self._sample_script):
                    raise AssertionError(
                        f"_SpecStageSim: sample_script exhausted "
                        f"(cursor={self._sample_cursor}, len="
                        f"{len(self._sample_script)})"
                    )
                next_token_id, is_terminal = self._sample_script[
                    self._sample_cursor
                ]
                self._sample_cursor += 1

        response = RunLayerSliceResponse.sign(
            identity=self.identity,
            request_id=request.request_id,
            activation_blob=out_blob,
            activation_shape=out_shape,
            activation_dtype=out_dtype,
            duration_seconds=0.001,
            tee_attestation=b"\x02" * 32,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
            next_token_id=next_token_id,
            is_terminal=is_terminal,
            verified_token_ids=verified_token_ids,
            accepted_count=accepted_count,
            # sp950 — mirror the real LayerStageServer: bind the per-stage input
            # commitment so the executor's (now-correct) anchor verification of
            # non-PREFILL stages succeeds. Without this the sim modelled the
            # pre-sp950 buggy contract (executor expected None) rather than the
            # real server.
            input_commitment=response_input_commitment_for_request(request),
        )
        return encode_message(response)


class _SpecTransport:
    def __init__(self, sims: Dict[str, _SpecStageSim]) -> None:
        self.sims = sims

    def send(self, address: str, request_bytes: bytes) -> bytes:
        sim = self.sims.get(address)
        if sim is None:
            raise ConnectionError(f"no sim at {address!r}")
        return sim.handle(request_bytes)


class _FakeDraft:
    """Fake DraftModel — proposes a scripted list of tokens per
    propose call. Records every call so tests can assert on
    parent_token_id threading + commit ordering."""

    def __init__(
        self, *, propose_script: List[List[int]],
    ) -> None:
        self._propose_script = list(propose_script)
        self._propose_cursor = 0
        self.reset_calls: List[dict] = []
        self.propose_calls: List[dict] = []
        self.commit_calls: List[dict] = []
        self.evict_calls: List[str] = []

    def reset(
        self, *, request_id: str, prompt_input_ids: List[int],
    ) -> None:
        self.reset_calls.append(
            {"request_id": request_id, "prompt": list(prompt_input_ids)}
        )

    def propose(
        self,
        *,
        request_id: str,
        parent_token_id: int,
        k: int,
        temperature: float,
    ) -> List[int]:
        self.propose_calls.append({
            "request_id": request_id,
            "parent_token_id": parent_token_id,
            "k": k,
            "temperature": temperature,
        })
        if self._propose_cursor >= len(self._propose_script):
            raise AssertionError(
                f"_FakeDraft: propose_script exhausted (cursor="
                f"{self._propose_cursor})"
            )
        out = self._propose_script[self._propose_cursor]
        self._propose_cursor += 1
        return list(out)

    def commit(
        self, *, request_id: str, accepted_token_ids: List[int],
    ) -> None:
        self.commit_calls.append({
            "request_id": request_id,
            "accepted": list(accepted_token_ids),
        })

    def evict(self, *, request_id: str) -> None:
        self.evict_calls.append(request_id)


class _RollbackLog:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, RollbackCacheRequest]] = []

    def __call__(self, address: str, payload: bytes) -> bytes:
        msg = parse_message(payload)
        assert isinstance(msg, RollbackCacheRequest)
        self.calls.append((address, msg))
        return encode_message(
            RollbackCacheResponse(
                request_id=msg.request_id,
                rolled_back=True,
                actual_dropped=msg.n_positions_to_drop,
            )
        )


class _EvictionLog:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, address: str, payload: bytes) -> bytes:
        msg = parse_message(payload)
        assert isinstance(msg, EvictCacheRequest)
        self.calls.append(address)
        return encode_message(
            EvictCacheResponse(
                request_id=msg.request_id, evicted=True,
            )
        )


def _prompt_passthrough(prompt: str) -> np.ndarray:
    raw = prompt.encode("utf-8")
    pad = (4 - len(raw) % 4) % 4
    return np.frombuffer(raw + b"\x00" * pad, dtype=np.int32).copy()


def _output_passthrough(arr: np.ndarray) -> str:
    return arr.tobytes().rstrip(b"\x00").decode("utf-8", errors="replace")


def _make_chain(stages: List[str], total_layers: int = 4) -> GPUChain:
    n = len(stages)
    per_stage = total_layers // n if n > 0 else 0
    layer_ranges = []
    for i in range(n):
        start = i * per_stage
        end = (i + 1) * per_stage if i < n - 1 else total_layers
        layer_ranges.append((start, end))
    return GPUChain(
        request_id="req-1",
        region="us-east",
        stages=tuple(stages),
        layer_ranges=tuple(layer_ranges),
        total_latency_ms=10.0,
        stale_profile_count=0,
    )


def _make_request(
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    prompt: str = "hello",
) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        model_id="m",
        budget_ftns=Decimal("10.0"),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        request_id="req-1",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _make_spec_executor(
    *,
    transport: _SpecTransport,
    settler,
    anchor,
    draft: _FakeDraft,
    speculation_depth: int = 4,
    rollback_send=None,
    cache_evict_send=None,
    encrypted_probs_cipher=None,
    flat_k_mode: bool = False,
    always_rollback_k: bool = False,
    per_stage_dispatch_cadence_seconds=None,
) -> RpcChainExecutor:
    return RpcChainExecutor(
        settler_identity=settler,
        send_message=transport.send,
        anchor=anchor,
        prompt_encoder=_prompt_passthrough,
        output_decoder=_output_passthrough,
        enable_sharded_decode=True,
        tokenizer=_FakeTokenizer(),
        draft_model=draft,
        speculation_depth=speculation_depth,
        rollback_cache_send_message=rollback_send,
        cache_evict_send_message=cache_evict_send,
        sharded_default_max_tokens=64,
        encrypted_probs_cipher=encrypted_probs_cipher,
        flat_k_mode=flat_k_mode,
        always_rollback_k=always_rollback_k,
        per_stage_dispatch_cadence_seconds=(
            per_stage_dispatch_cadence_seconds
        ),
    )


def _build_single_stage(
    *,
    sample_script: List[Tuple[int, bool]],
    verify_script: List[Tuple[Tuple[int, ...], int, bool]],
):
    settler = generate_node_identity()
    tail = generate_node_identity()
    anchor = _FakeAnchor()
    anchor.registered[tail.node_id] = tail.public_key_b64
    sim = _SpecStageSim(
        tail,
        is_tail=True,
        sample_script=sample_script,
        verify_script=verify_script,
    )
    transport = _SpecTransport({tail.node_id: sim})
    chain = _make_chain([tail.node_id])
    return settler, anchor, transport, sim, chain


def _build_two_stage(
    *,
    sample_script: List[Tuple[int, bool]],
    verify_script: List[Tuple[Tuple[int, ...], int, bool]],
):
    settler = generate_node_identity()
    s1 = generate_node_identity()
    s2 = generate_node_identity()
    anchor = _FakeAnchor()
    anchor.registered[s1.node_id] = s1.public_key_b64
    anchor.registered[s2.node_id] = s2.public_key_b64
    sim1 = _SpecStageSim(s1, is_tail=False)
    sim2 = _SpecStageSim(
        s2, is_tail=True,
        sample_script=sample_script,
        verify_script=verify_script,
    )
    transport = _SpecTransport({
        s1.node_id: sim1, s2.node_id: sim2,
    })
    chain = _make_chain([s1.node_id, s2.node_id])
    return settler, anchor, transport, (sim1, sim2), chain


# ──────────────────────────────────────────────────────────────────────────
# Construction validation
# ──────────────────────────────────────────────────────────────────────────


class TestSpeculativeConstructionValidation:
    def test_draft_requires_sharded_decode(self):
        settler = generate_node_identity()
        anchor = _FakeAnchor()
        with pytest.raises(
            RuntimeError, match="enable_sharded_decode=True"
        ):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_passthrough,
                output_decoder=_output_passthrough,
                enable_sharded_decode=False,  # missing
                draft_model=_FakeDraft(propose_script=[[1]]),
            )

    def test_draft_missing_methods_rejected(self):
        settler = generate_node_identity()
        anchor = _FakeAnchor()

        class _BadDraft:
            def reset(self, **kw): ...
            # missing propose/commit/evict

        with pytest.raises(RuntimeError, match="propose"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_passthrough,
                output_decoder=_output_passthrough,
                enable_sharded_decode=True,
                tokenizer=_FakeTokenizer(),
                draft_model=_BadDraft(),
            )

    def test_speculation_depth_bool_rejected(self):
        settler = generate_node_identity()
        anchor = _FakeAnchor()
        with pytest.raises(ValueError, match="speculation_depth"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_passthrough,
                output_decoder=_output_passthrough,
                enable_sharded_decode=True,
                tokenizer=_FakeTokenizer(),
                draft_model=_FakeDraft(propose_script=[[1]]),
                speculation_depth=True,  # type: ignore[arg-type]
            )

    def test_speculation_depth_zero_rejected(self):
        settler = generate_node_identity()
        anchor = _FakeAnchor()
        with pytest.raises(ValueError, match="speculation_depth"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_passthrough,
                output_decoder=_output_passthrough,
                enable_sharded_decode=True,
                tokenizer=_FakeTokenizer(),
                draft_model=_FakeDraft(propose_script=[[1]]),
                speculation_depth=0,
            )

    def test_speculation_depth_above_cap_rejected(self):
        from prsm.compute.chain_rpc.protocol import (
            MAX_VERIFY_BATCH_TOKENS,
        )
        settler = generate_node_identity()
        anchor = _FakeAnchor()
        with pytest.raises(ValueError, match="exceeds K cap"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_passthrough,
                output_decoder=_output_passthrough,
                enable_sharded_decode=True,
                tokenizer=_FakeTokenizer(),
                draft_model=_FakeDraft(propose_script=[[1]]),
                speculation_depth=MAX_VERIFY_BATCH_TOKENS,
            )

    def test_non_callable_rollback_rejected(self):
        settler = generate_node_identity()
        anchor = _FakeAnchor()
        with pytest.raises(
            RuntimeError, match="rollback_cache_send_message"
        ):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_passthrough,
                output_decoder=_output_passthrough,
                enable_sharded_decode=True,
                tokenizer=_FakeTokenizer(),
                draft_model=_FakeDraft(propose_script=[[1]]),
                rollback_cache_send_message="not callable",  # type: ignore[arg-type]
            )

    def test_temperature_nonzero_without_v2_capability_rejected(self):
        # Phase 3.x.11.y.x Task 5: greedy-only gate is LIFTED — v2
        # stochastic path now permitted IFF draft_model exposes
        # propose_with_probs. Drafts without v2 capability still
        # raise at temperature > 0 (graceful capability error,
        # not a hard "greedy-only" rejection).
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(42, False)],
            verify_script=[],
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            # _FakeDraft only implements v1 propose; no
            # propose_with_probs attribute.
            draft=_FakeDraft(propose_script=[[100]]),
        )
        gen = executor.execute_chain_streaming(
            request=_make_request(max_tokens=4, temperature=0.7),
            chain=chain,
        )
        with pytest.raises(
            ChainExecutionError, match="propose_with_probs"
        ):
            next(gen)


# ──────────────────────────────────────────────────────────────────────────
# Speculation behaviors
# ──────────────────────────────────────────────────────────────────────────


class TestSpeculationLoop:
    def test_full_accept_emits_k_plus_one_per_round(self):
        # K=4. PREFILL emits token 7. Round 1: drafts [10,20,30,40];
        # verifier returns [10,20,30,40,99] all-match → emit
        # 5 tokens (round 1 alone produces tokens 8..12). Total 6
        # tokens emitted. max_tokens=6 → terminate after round 1.
        settler, anchor, transport, sim, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 30, 40, 99), 4, False)],
        )
        rollback_log = _RollbackLog()
        evict_log = _EvictionLog()
        draft = _FakeDraft(propose_script=[[10, 20, 30, 40]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=4,
            rollback_send=rollback_log, cache_evict_send=evict_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=6, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 10, 20, 30, 40, 99]
        # All-accepted: rollback_count = K+1 - (K+1) = 0 → no
        # broadcast.
        assert rollback_log.calls == []
        # Eviction broadcast on terminal.
        assert len(evict_log.calls) == 1
        # Draft model received reset, propose, commit, evict.
        assert len(draft.reset_calls) == 1
        assert len(draft.propose_calls) == 1
        assert draft.propose_calls[0]["parent_token_id"] == 7
        assert draft.propose_calls[0]["k"] == 4
        assert len(draft.commit_calls) == 1
        assert draft.commit_calls[0]["accepted"] == [10, 20, 30, 40, 99]
        assert draft.evict_calls == ["req-1"]

    def test_zero_accepted_emits_one_correction(self):
        # PREFILL emits token 7. Round 1 drafts [10, 20]; verifier
        # returns [50, 51, 52], 0 match → emit only [50]. Round 2
        # drafts [60, 61]; verifier returns [70, 71, 72], 0 match
        # → emit [70]. max_tokens=3 → terminate.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[
                ((50, 51, 52), 0, False),
                ((70, 71, 72), 0, False),
            ],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraft(propose_script=[[10, 20], [60, 61]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            rollback_send=rollback_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=3, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 50, 70]
        # Each round rolled back K+1 - 1 = 2 positions.
        assert len(rollback_log.calls) == 2
        for _, msg in rollback_log.calls:
            assert msg.n_positions_to_drop == 2

    def test_partial_accept(self):
        # PREFILL emits 7. K=4, drafts [10, 20, 30, 40]; verifier
        # returns [10, 20, 99, 98, 97], accepted_count=2 → emit
        # [10, 20, 99]. max_tokens=4 → next round drafts [50,51,
        # 52,53]; verifier returns [200, ...], 0 match → emit
        # [200]. Total 5 tokens (PREFILL + 4 = max).
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[
                ((10, 20, 99, 98, 97), 2, False),
                ((200, 201, 202, 203, 204), 0, False),
            ],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraft(propose_script=[
            [10, 20, 30, 40],
            [50, 51, 52, 53],
        ])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=4,
            rollback_send=rollback_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=5, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        # PREFILL=7, partial=10,20,99, then round 2's correction=200
        assert [t.token_id for t in tokens] == [7, 10, 20, 99, 200]
        # Round 1: rolled back K+1 - (accepted+1) = 5 - 3 = 2.
        # Round 2: rolled back K+1 - 1 = 4.
        assert [m.n_positions_to_drop for _, m in rollback_log.calls] == [2, 4]
        # Round 2's parent = last emitted from round 1 = 99.
        assert draft.propose_calls[1]["parent_token_id"] == 99

    def test_eos_in_emitted_terminates(self):
        # PREFILL emits 7. K=4, drafts [10, 20, 30, 40]; verifier
        # returns [10, 20, 999_eos, 98, 97], accepted_count=2 →
        # emit [10, 20, 999]. Tail flags terminal=True. Loop ends.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 999, 98, 97), 2, True)],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraft(propose_script=[[10, 20, 30, 40]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=4,
            rollback_send=rollback_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=64, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 10, 20, 999]
        assert tokens[-1].finish_reason == "stop"

    def test_max_tokens_cap_truncates_mid_round(self):
        # max_tokens=3. PREFILL=7. Round 1 drafts [10, 20, 30, 40];
        # verifier returns full-accept [10,20,30,40,99], so
        # emit-cap = 3 - 1 (already emitted PREFILL) = 2 → emit
        # [10, 20] only. Round 1 should TRUNCATE.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 30, 40, 99), 4, False)],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraft(propose_script=[[10, 20, 30, 40]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=4,
            rollback_send=rollback_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=3, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 10, 20]
        assert tokens[-1].finish_reason == "max_tokens"
        # 5 cached - 2 emitted = 3 dropped via rollback.
        assert len(rollback_log.calls) == 1
        assert rollback_log.calls[0][1].n_positions_to_drop == 3
        # draft.commit was called with the actually-emitted prefix.
        assert draft.commit_calls[0]["accepted"] == [10, 20]

    def test_rollback_broadcast_only_on_partial_accept(self):
        # All-accepted round → no rollback. Partial round →
        # rollback with the correct count.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[
                ((10, 20, 30), 2, False),  # K=2, all match → 0 dropped
                ((40, 41, 42), 0, False),  # K=2, 0 match → 2 dropped
            ],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraft(propose_script=[[10, 20], [50, 51]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            rollback_send=rollback_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=5, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        # PREFILL=7, round1=[10,20,30] (all 3 emitted), round2=[40]
        assert [t.token_id for t in tokens] == [7, 10, 20, 30, 40]
        # Only round 2 triggered rollback (round 1 was full-accept
        # → no extra cached).
        assert len(rollback_log.calls) == 1
        assert rollback_log.calls[0][1].n_positions_to_drop == 2

    def test_two_stage_chain_threads_proposed_to_tail(self):
        # 2-stage chain. Stage 1 sees VERIFY input as 5 token ids
        # in activation_blob; stage 2 (tail) computes
        # accepted_count. Verify the wire request to BOTH stages
        # carries proposed_token_ids.
        settler, anchor, transport, sims, chain = _build_two_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 30, 40, 99), 4, False)],
        )
        sim1, sim2 = sims
        draft = _FakeDraft(propose_script=[[10, 20, 30, 40]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=4,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=6, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 10, 20, 30, 40, 99]
        # Find the VERIFY request hitting each stage.
        verify_reqs_s1 = [
            r for r in sim1.requests
            if r.decode_mode == DecodeMode.VERIFY
        ]
        verify_reqs_s2 = [
            r for r in sim2.requests
            if r.decode_mode == DecodeMode.VERIFY
        ]
        assert len(verify_reqs_s1) == 1
        assert len(verify_reqs_s2) == 1
        assert verify_reqs_s1[0].proposed_token_ids == (10, 20, 30, 40)
        assert verify_reqs_s2[0].proposed_token_ids == (10, 20, 30, 40)

    def test_cancellation_evicts_draft_and_cache(self):
        # Caller closes the generator after the first yield. finally
        # block must call draft.evict + eviction broadcast.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 99), 2, False)] * 10,
        )
        evict_log = _EvictionLog()
        draft = _FakeDraft(
            propose_script=[[10, 20]] * 10,
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            cache_evict_send=evict_log,
        )
        gen = executor.execute_chain_streaming(
            request=_make_request(max_tokens=64, temperature=0.0),
            chain=chain,
        )
        # Pull the first token (PREFILL emit) then close mid-stream.
        first = next(gen)
        assert isinstance(first, StreamToken)
        assert first.token_id == 7
        gen.close()
        # Eviction fired despite early close.
        assert len(evict_log.calls) == 1
        assert draft.evict_calls == ["req-1"]

    def test_greedy_equivalence_with_non_speculative(self):
        # With temperature=0, the speculative output (as token id
        # sequence) MUST match a non-speculative single-token
        # decode for the same prompt. Construct two parallel
        # single-stage runs:
        #   A) speculative, K=4, drafts always all-accept against
        #      verifier's [50, 60, 70, 80, 90] script.
        #   B) non-speculative — sample_script returns
        #      [50, 60, 70, 80, 90, 99] (one per call).
        # Both should emit the same first 6 tokens (PREFILL=42
        # in A's sample_script vs B's first sample, plus 5 from
        # spec/incremental). To make them comparable, both runs
        # use sample_script that produces 42 on PREFILL.

        # Run A (speculative).
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(42, False)],
            verify_script=[((50, 60, 70, 80, 90), 4, False)],
        )
        draft = _FakeDraft(propose_script=[[50, 60, 70, 80]])
        spec_exec = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=4,
        )
        spec_events = list(spec_exec.execute_chain_streaming(
            request=_make_request(max_tokens=6, temperature=0.0),
            chain=chain,
        ))
        spec_ids = [
            e.token_id for e in spec_events
            if isinstance(e, StreamToken)
        ]
        # Run B (non-speculative — fresh executor + transport).
        settler_b = generate_node_identity()
        tail_b = generate_node_identity()
        anchor_b = _FakeAnchor()
        anchor_b.registered[tail_b.node_id] = tail_b.public_key_b64
        sim_b = _SpecStageSim(
            tail_b, is_tail=True,
            sample_script=[
                (42, False), (50, False), (60, False),
                (70, False), (80, False), (90, False),
            ],
        )
        trans_b = _SpecTransport({tail_b.node_id: sim_b})
        chain_b = _make_chain([tail_b.node_id])
        baseline_exec = RpcChainExecutor(
            settler_identity=settler_b,
            send_message=trans_b.send,
            anchor=anchor_b,
            prompt_encoder=_prompt_passthrough,
            output_decoder=_output_passthrough,
            enable_sharded_decode=True,
            tokenizer=_FakeTokenizer(),
            sharded_default_max_tokens=64,
        )
        baseline_events = list(baseline_exec.execute_chain_streaming(
            request=_make_request(max_tokens=6, temperature=0.0),
            chain=chain_b,
        ))
        baseline_ids = [
            e.token_id for e in baseline_events
            if isinstance(e, StreamToken)
        ]
        assert spec_ids == baseline_ids == [42, 50, 60, 70, 80, 90]

    def test_chain_terminal_overrides_continuing_speculation(self):
        # Tail flagged is_terminal=True on first VERIFY round →
        # loop exits after that round even if max_tokens budget
        # remains.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 30), 2, True)],  # all match + EOS
        )
        draft = _FakeDraft(propose_script=[[10, 20]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=64, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        # PREFILL + 3 emitted (full accept includes bonus); loop
        # terminated by chain's is_terminal=True.
        assert [t.token_id for t in tokens] == [7, 10, 20, 30]
        # Only one propose round despite generous max_tokens.
        assert len(draft.propose_calls) == 1


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.y.x Task 5 — v2 (Leviathan-2023 stochastic) routing
# ──────────────────────────────────────────────────────────────────────────


class _FakeDraftV2(_FakeDraft):
    """v2-capable draft. Adds ``propose_with_probs`` returning
    ``(ids, probs)`` from a parallel script. Falls back to v1
    propose for the v1 path (not exercised when temperature > 0)."""

    def __init__(
        self,
        *,
        propose_script: List[List[int]],
        probs_script: List[List[float]],
    ) -> None:
        super().__init__(propose_script=propose_script)
        if len(probs_script) != len(propose_script):
            raise AssertionError(
                "_FakeDraftV2: probs_script must align 1:1 with "
                "propose_script"
            )
        self._probs_script = list(probs_script)
        self._probs_cursor = 0
        self.propose_with_probs_calls: List[dict] = []

    def propose_with_probs(
        self,
        *,
        request_id: str,
        parent_token_id: int,
        k: int,
        temperature: float,
    ) -> Tuple[List[int], List[float]]:
        self.propose_with_probs_calls.append({
            "request_id": request_id,
            "parent_token_id": parent_token_id,
            "k": k,
            "temperature": temperature,
        })
        if self._propose_cursor >= len(self._propose_script):
            raise AssertionError(
                f"_FakeDraftV2: script exhausted (cursor="
                f"{self._propose_cursor})"
            )
        ids = list(self._propose_script[self._propose_cursor])
        probs = list(self._probs_script[self._probs_cursor])
        self._propose_cursor += 1
        self._probs_cursor += 1
        return ids, probs


class TestV2StochasticRoutingExecutor:
    """v1 stays bit-identical (regression covered above by
    TestSpeculationLoop). These tests cover the new v2 (temperature
    > 0) path: propose_with_probs is called, probs ride the wire,
    accepted_count + 1 verified entries are emitted, rollback math
    accounts for the K+1 cached vs accepted+1 emitted asymmetry,
    and adaptive K halves/doubles on the rolling accept-rate."""

    def test_v2_partial_accept_emits_accepted_plus_one(self):
        # K=2; v2 returns accepted_count=1 with verified=(11, 50)
        # — the second draft was rejected, 50 is the rejection-
        # sampled correction. Length is 2 (= ac + 1), NOT K+1=3.
        settler, anchor, transport, sim, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((11, 50), 1, False)],  # v2 shape
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraftV2(
            propose_script=[[10, 20]],
            probs_script=[[0.6, 0.4]],
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            rollback_send=rollback_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=3, temperature=0.7),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 11, 50]
        # propose_with_probs invoked, NOT propose.
        assert len(draft.propose_with_probs_calls) == 1
        assert draft.propose_with_probs_calls[0]["temperature"] == 0.7
        assert len(draft.propose_calls) == 0
        # Rollback: K+1 cached - len(emitted) = 3 - 2 = 1. The
        # critical correctness check vs v1 — len(verified)=2 in v2,
        # if rollback math used len(verified) it would compute 0
        # and miss the rejected draft's stale cache entry.
        assert len(rollback_log.calls) == 1
        assert rollback_log.calls[0][1].n_positions_to_drop == 1
        # Wire-side: probs rode through to the stage.
        verify_req = next(
            r for r in sim.requests
            if r.decode_mode == DecodeMode.VERIFY
        )
        assert verify_req.proposed_token_ids == (10, 20)
        assert verify_req.proposed_token_probs == (0.6, 0.4)

    def test_v2_full_accept_no_rollback(self):
        # accepted_count == K → verified has K+1 entries (last is
        # bonus). Rollback math: 3 - 3 = 0, no broadcast.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 99), 2, False)],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraftV2(
            propose_script=[[10, 20]],
            probs_script=[[0.9, 0.9]],
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            rollback_send=rollback_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=4, temperature=0.7),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 10, 20, 99]
        assert rollback_log.calls == []

    def test_v2_zero_accepted_one_correction(self):
        # accepted_count == 0 → verified=(corr,) length 1.
        # Rollback: K+1 - 1 = K positions stale.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((50,), 0, False)],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraftV2(
            propose_script=[[10, 20, 30]],
            probs_script=[[0.5, 0.5, 0.5]],
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=3,
            rollback_send=rollback_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=2, temperature=0.7),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 50]
        assert rollback_log.calls[0][1].n_positions_to_drop == 3

    def test_v2_max_tokens_cap_mid_emit(self):
        # max_tokens=2: PREFILL emits 1; round 1 v2 emits up to 2
        # but cap allows only 1 more → emitted = [11], rollback
        # = K+1 - 1 = 2.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((11, 12, 99), 2, False)],  # full accept
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraftV2(
            propose_script=[[11, 12]],
            probs_script=[[0.9, 0.9]],
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            rollback_send=rollback_log,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=2, temperature=0.7),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 11]
        assert tokens[-1].finish_reason == "max_tokens"
        # K+1 cached = 3; emitted = 1; rollback = 2.
        assert rollback_log.calls[0][1].n_positions_to_drop == 2

    def test_v2_temperature_zero_keeps_v1_path(self):
        # Even when draft has propose_with_probs, temperature=0.0
        # MUST take the v1 (greedy) path — preserves bit-identical
        # greedy regression.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 99), 2, False)],  # K+1 v1 shape
        )
        draft = _FakeDraftV2(
            propose_script=[[10, 20]],
            probs_script=[[0.9, 0.9]],
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=4, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 10, 20, 99]
        # v1 path → propose called, propose_with_probs NOT.
        assert len(draft.propose_calls) == 1
        assert len(draft.propose_with_probs_calls) == 0

    def test_v2_rejects_malformed_probs_length(self):
        # propose_with_probs returns mismatched lengths → raise
        # MALFORMED_RESPONSE.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[],
        )
        draft = _FakeDraftV2(
            propose_script=[[10, 20]],
            probs_script=[[0.5]],   # length 1 vs 2 ids
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
        )
        gen = executor.execute_chain_streaming(
            request=_make_request(max_tokens=4, temperature=0.7),
            chain=chain,
        )
        with pytest.raises(
            ChainExecutionError, match="co-set with equal length"
        ):
            list(gen)

    def test_v2_tail_v1_shape_at_temp_gt_zero_raises(self):
        # Server speaking v1 (returning K+1 entries) under v2
        # request → reject as MALFORMED_RESPONSE. Catches a
        # backwards-compat hole: a stale tail must not silently
        # interpret v2 stochastic dispatch as v1 greedy.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 99), 1, False)],  # ac=1 but K+1=3 entries
        )
        draft = _FakeDraftV2(
            propose_script=[[10, 20]],
            probs_script=[[0.6, 0.4]],
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
        )
        gen = executor.execute_chain_streaming(
            request=_make_request(max_tokens=4, temperature=0.7),
            chain=chain,
        )
        with pytest.raises(
            ChainExecutionError, match="v2 stochastic"
        ):
            list(gen)


class TestAdaptiveK:
    """Phase 3.x.11.y.x Task 5 — rolling-window adaptive K."""

    def test_adaptive_k_doubles_on_high_accept_rate(self):
        # 4 rounds of full accept (K=2, ac=2 each → rate = 1.0).
        # After window fills (round 4), next-round K should
        # double from 2 → 4. Round 5 propose call must request
        # k=4. Each full-accept round emits ac+1=3 tokens, so
        # max_tokens=20 keeps loop running long enough for at
        # least one post-adapt round.
        sample_script = [(7, False)]
        verify_script = [
            ((10, 20, 99), 2, False),
            ((30, 40, 98), 2, False),
            ((50, 60, 97), 2, False),
            ((70, 80, 96), 2, False),
            # Round 5 must use k=4 (verified has 5 = K+1 entries).
            ((90, 91, 92, 93, 94), 4, True),
        ]
        propose_script = [
            [10, 20], [30, 40], [50, 60], [70, 80],
            [90, 91, 92, 93],
        ]
        probs_script = [[0.9] * len(p) for p in propose_script]
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=sample_script,
            verify_script=verify_script,
        )
        draft = _FakeDraftV2(
            propose_script=propose_script,
            probs_script=probs_script,
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
        )
        list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=20, temperature=0.7),
            chain=chain,
        ))
        # First 4 rounds at K=2; round 5 at K=4 (doubled).
        ks = [c["k"] for c in draft.propose_with_probs_calls]
        assert ks == [2, 2, 2, 2, 4]

    def test_adaptive_k_halves_on_low_accept_rate(self):
        # 4 rounds K=4 with ac=0 each (rate 0%). Round 5 K = 4//2
        # = 2.
        sample_script = [(7, False)]
        verify_script = [
            ((50,), 0, False),
            ((51,), 0, False),
            ((52,), 0, False),
            ((53,), 0, False),
            ((54,), 0, True),
        ]
        propose_script = [
            [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18],   # K halved to 2
        ]
        probs_script = [[0.5] * len(p) for p in propose_script]
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=sample_script,
            verify_script=verify_script,
        )
        draft = _FakeDraftV2(
            propose_script=propose_script,
            probs_script=probs_script,
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=4,
        )
        list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=20, temperature=0.7),
            chain=chain,
        ))
        ks = [c["k"] for c in draft.propose_with_probs_calls]
        assert ks == [4, 4, 4, 4, 2]

    def test_adaptive_k_holds_in_neutral_band(self):
        # 4 rounds K=4 with ac=2 each (rate 50%). K stays at 4.
        sample_script = [(7, False)]
        verify_script = [
            ((10, 20, 50), 2, False),
            ((30, 40, 51), 2, False),
            ((60, 70, 52), 2, False),
            ((80, 90, 53), 2, False),
            ((100, 110, 54), 2, True),
        ]
        # Each round: K=4 drafts but only 2 accepted; verified is
        # ac+1 = 3 entries. Round 5 must still use K=4.
        propose_script = [
            [10, 20, 30, 40], [30, 40, 41, 42],
            [60, 70, 71, 72], [80, 90, 91, 92],
            [100, 110, 111, 112],
        ]
        probs_script = [[0.5] * 4 for _ in propose_script]
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=sample_script,
            verify_script=verify_script,
        )
        draft = _FakeDraftV2(
            propose_script=propose_script,
            probs_script=probs_script,
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=4,
        )
        list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=20, temperature=0.7),
            chain=chain,
        ))
        ks = [c["k"] for c in draft.propose_with_probs_calls]
        assert ks == [4, 4, 4, 4, 4]

    def test_adaptive_k_caps_at_max_verify_batch(self):
        # K = MAX-1 already; high accept rate → wants to double,
        # but stays capped at MAX_VERIFY_BATCH_TOKENS - 1.
        from prsm.compute.chain_rpc.protocol import (
            MAX_VERIFY_BATCH_TOKENS,
        )
        k_max = MAX_VERIFY_BATCH_TOKENS - 1
        # Use modest k_max (cap won't blow test runtime — verified
        # tuples scale with K). For tractable test, run K=4 with
        # speculation_depth=k_max-1, force doubling to hit cap.
        # Simpler: speculation_depth=k_max and run 4 full-accept
        # rounds; round 5 should still be k_max.
        sample_script = [(7, False)]
        verify_script = []
        propose_script = []
        probs_script = []
        for _ in range(5):
            verified = tuple(range(100, 100 + k_max + 1))
            verify_script.append((verified, k_max, False))
            propose_script.append([200 + i for i in range(k_max)])
            probs_script.append([0.99] * k_max)
        # Final round terminate.
        verify_script[-1] = (verify_script[-1][0], k_max, True)
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=sample_script,
            verify_script=verify_script,
        )
        draft = _FakeDraftV2(
            propose_script=propose_script,
            probs_script=probs_script,
        )
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=k_max,
        )
        list(executor.execute_chain_streaming(
            request=_make_request(
                max_tokens=k_max * 10, temperature=0.7,
            ),
            chain=chain,
        ))
        ks = [c["k"] for c in draft.propose_with_probs_calls]
        # All five rounds at k_max — capped, no overshoot.
        assert ks == [k_max] * 5


class TestAlwaysRollbackK:
    """Phase 3.x.11.q.y' — always-rollback-K + replay-prefix
    protocol. Closes the residual rollback drop-value leak from
    threat-model §3.8 by dispatching a constant-K rollback per
    VERIFY round regardless of accepted_count, accompanied by the
    accepted prefix tokens (encrypted under the wired cipher when
    present)."""

    def test_constant_k_rollback_on_full_accept(self):
        # Full-accept round: v1 path emits NO rollback (cached_extra
        # == 0). Always-rollback-K mode emits a rollback with
        # n_positions_to_drop == K + 1 anyway.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 99), 2, False)],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraft(propose_script=[[10, 20]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            rollback_send=rollback_log,
            always_rollback_k=True,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=4, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        # PREFILL=7, full-accept K=2 + bonus → emit [10, 20, 99]
        assert [t.token_id for t in tokens] == [7, 10, 20, 99]
        # Always-rollback-K dispatched a rollback with K+1 dropped
        # even though accepted_count == K (full accept).
        assert len(rollback_log.calls) == 1
        assert rollback_log.calls[0][1].n_positions_to_drop == 3  # K+1
        # replay_accepted_prefix carries the actually-emitted tokens
        # so the server can replay them after the constant-K
        # truncation rebuilds the cache to the correct state.
        assert rollback_log.calls[0][1].replay_accepted_prefix == (
            10, 20, 99,
        )

    def test_constant_k_rollback_on_partial_accept(self):
        # Partial accept (1 of K=2 + bonus mismatch). v1 path drops
        # 1; q.y' drops K+1 = 3 with prefix carrying the 2 emitted.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 99, 88), 1, False)],  # partial: 1 ac
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraft(propose_script=[[10, 20]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            rollback_send=rollback_log,
            always_rollback_k=True,
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=3, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        # PREFILL=7, ac=1 + correction → emit [10, 99]; loop ends
        # at max_tokens=3.
        assert [t.token_id for t in tokens] == [7, 10, 99]
        assert len(rollback_log.calls) == 1
        # CONSTANT-K invariant: drop count is K+1 regardless of
        # actual accepted_count. Wire observer cannot distinguish
        # this round (ac=1) from the full-accept round (ac=K)
        # purely from the rollback envelope.
        assert rollback_log.calls[0][1].n_positions_to_drop == 3
        # Prefix has only the actually-emitted 2 tokens.
        assert rollback_log.calls[0][1].replay_accepted_prefix == (
            10, 99,
        )

    def test_v1_mode_unchanged(self):
        # Default (always_rollback_k=False) preserves v1 behavior:
        # full accept → no rollback; partial accept → drop count
        # equals (K+1) - len(emitted).
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[
                ((10, 20, 99), 2, False),    # full: 0 dropped
                ((40, 41, 42), 0, False),    # 0 ac: 2 dropped
            ],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraft(propose_script=[[10, 20], [50, 51]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            rollback_send=rollback_log,
            # always_rollback_k defaults to False.
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=5, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert [t.token_id for t in tokens] == [7, 10, 20, 99, 40]
        # v1: only round 2 triggered rollback.
        assert len(rollback_log.calls) == 1
        assert rollback_log.calls[0][1].n_positions_to_drop == 2
        # v1 doesn't carry the replay prefix (backwards-compat).
        assert (
            rollback_log.calls[0][1].replay_accepted_prefix is None
        )
        assert (
            rollback_log.calls[0][1].encrypted_replay_accepted_prefix
            is None
        )

    def test_encrypted_prefix_when_cipher_wired(self):
        # When always_rollback_k=True AND encrypted_probs_cipher is
        # wired, the prefix is encrypted on the wire (plaintext
        # field is None, encrypted field is set).
        from prsm.compute.chain_rpc.probs_cipher import (
            ProbsCipher,
            derive_key_from_psk,
        )

        cipher = ProbsCipher(
            key=derive_key_from_psk(b"test-psk-q-y-prime-rollback"),
        )
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 99), 2, False)],
        )
        rollback_log = _RollbackLog()
        draft = _FakeDraft(propose_script=[[10, 20]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            rollback_send=rollback_log,
            encrypted_probs_cipher=cipher,
            flat_k_mode=True,
            always_rollback_k=True,
        )
        list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=4, temperature=0.0),
            chain=chain,
        ))
        assert len(rollback_log.calls) == 1
        rb = rollback_log.calls[0][1]
        # Encrypted form on the wire; plaintext field is None.
        assert rb.encrypted_replay_accepted_prefix is not None
        assert rb.replay_accepted_prefix is None
        # Cipher must round-trip back to the original prefix.
        decrypted = cipher.decrypt_prefix(
            ciphertext=rb.encrypted_replay_accepted_prefix,
            request_id=rb.request_id,
            stage_index=0,
            expected_k=3,
        )
        assert decrypted == [10, 20, 99]


class TestPerStageDispatchCadence:
    """Phase 3.x.11.q.x — per_stage_dispatch_cadence_seconds
    constructor flag + sharded-loop integration. Closes the §7.13
    honest-scope item #1 (per-stage wire still leaks per-token
    timing under sharded autoregressive decode)."""

    def test_constructor_rejects_non_positive_cadence(self):
        # Validator: cadence must be positive when set.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 99), 2, False)],
        )
        draft = _FakeDraft(propose_script=[[10, 20]])
        for bad in [0, -0.5, -1]:
            with pytest.raises(RuntimeError, match="positive"):
                _make_spec_executor(
                    transport=transport, settler=settler, anchor=anchor,
                    draft=draft, speculation_depth=2,
                    per_stage_dispatch_cadence_seconds=bad,
                )

    def test_constructor_rejects_bool_cadence(self):
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 99), 2, False)],
        )
        draft = _FakeDraft(propose_script=[[10, 20]])
        with pytest.raises(RuntimeError, match="must be number"):
            _make_spec_executor(
                transport=transport, settler=settler, anchor=anchor,
                draft=draft, speculation_depth=2,
                per_stage_dispatch_cadence_seconds=True,  # type: ignore[arg-type]
            )

    def test_no_cadence_default_unaffected(self):
        # Default (None) preserves legacy behavior — no sleeps.
        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[((10, 20, 99), 2, False)],
        )
        draft = _FakeDraft(propose_script=[[10, 20]])
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            # per_stage_dispatch_cadence_seconds=None default
        )
        events = list(executor.execute_chain_streaming(
            request=_make_request(max_tokens=4, temperature=0.0),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        # PREFILL=7 + verify K=2 full-accept → 4 tokens (7,10,20,99).
        assert [t.token_id for t in tokens] == [7, 10, 20, 99]

    def test_cadence_clamps_inter_iteration_via_sleep_calls(self):
        # tests/conftest.py installs an auto-use mock that patches
        # time.sleep → instant (so unit tests don't actually
        # block). We can't use wall-clock to verify cadence; we
        # instead patch `time.sleep` ourselves to record calls
        # and assert at least one sleep with duration > 0
        # happened (= cadence clamp fired between iterations).
        import time as _time
        sleep_calls: List[float] = []
        original_sleep = _time.sleep

        def recording_sleep(seconds: float) -> None:
            sleep_calls.append(float(seconds))
            # Don't actually sleep — fast test.

        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[
                ((10, 20, 99), 2, False),  # round 1 full-accept
                ((30, 40, 88), 2, False),  # round 2 full-accept
            ],
        )
        draft = _FakeDraft(propose_script=[[10, 20], [30, 40]])
        cadence = 0.05  # 50ms
        executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            per_stage_dispatch_cadence_seconds=cadence,
        )
        # Replace time.sleep on the imported module that the
        # executor uses (chain_rpc.client imports `time` at
        # module level + calls `time.sleep` inside the helper).
        from prsm.compute.chain_rpc import client as _client_mod
        _client_mod.time.sleep = recording_sleep
        try:
            list(executor.execute_chain_streaming(
                request=_make_request(max_tokens=7, temperature=0.0),
                chain=chain,
            ))
        finally:
            _client_mod.time.sleep = original_sleep
        # At least one sleep call with a positive duration close
        # to the cadence. PREFILL → VERIFY round 1 inter-iter gap
        # is the first sleep; round 1 → round 2 is the second.
        positive_sleeps = [s for s in sleep_calls if s > 0]
        assert len(positive_sleeps) >= 1, (
            f"cadence clamp didn't fire — no positive sleep calls "
            f"recorded (sleep_calls={sleep_calls})"
        )
        # Each sleep should be ≤ cadence (helper sleeps the
        # remainder, not more).
        for s in positive_sleeps:
            assert s <= cadence + 1e-6, (
                f"sleep duration {s} > cadence {cadence} "
                f"(helper bug — sleeping more than cadence)"
            )


class TestQXCompositionCadencePlusPadding:
    """Phase 3.x.11.q.x Task 3 — composition smoke. The inner
    RpcChainExecutor enforces per-stage dispatch cadence; an
    outer BatchedTrailingShardedExecutor pads the trailing
    StreamToken to a fixed byte length. Both fire on the same
    request without interfering."""

    def test_cadence_and_padding_compose(self):
        from prsm.compute.chain_rpc.tier_c_sharded_executors import (
            BatchedTrailingShardedExecutor,
        )

        sleep_calls: List[float] = []

        def recording_sleep(seconds: float) -> None:
            sleep_calls.append(float(seconds))

        settler, anchor, transport, _, chain = _build_single_stage(
            sample_script=[(7, False)],
            verify_script=[
                ((10, 20, 99), 2, False),
                ((30, 40, 88), 2, False),
            ],
        )
        draft = _FakeDraft(propose_script=[[10, 20], [30, 40]])
        cadence = 0.05
        inner_executor = _make_spec_executor(
            transport=transport, settler=settler, anchor=anchor,
            draft=draft, speculation_depth=2,
            per_stage_dispatch_cadence_seconds=cadence,
        )
        # Wrap in M2 with pad_to_bytes.
        decorator = BatchedTrailingShardedExecutor(
            inner=inner_executor, pad_to_bytes=64,
        )
        from prsm.compute.chain_rpc import client as _client_mod
        original_sleep = _client_mod.time.sleep
        _client_mod.time.sleep = recording_sleep
        try:
            events = list(decorator.execute_chain_streaming(
                request=_make_request(
                    max_tokens=7, temperature=0.0,
                ),
                chain=chain,
            ))
        finally:
            _client_mod.time.sleep = original_sleep
        # Cadence fired (at least one positive sleep).
        positive_sleeps = [s for s in sleep_calls if s > 0]
        assert len(positive_sleeps) >= 1, (
            "cadence didn't fire under composition"
        )
        # M2 emitted exactly 1 StreamToken with padded length.
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert len(tokens) == 1
        assert len(tokens[0].text_delta.encode("utf-8")) == 64
