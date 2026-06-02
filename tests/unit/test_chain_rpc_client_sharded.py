"""Phase 3.x.11 Task 5 — RpcChainExecutor sharded-decode tests.

Covers the new ``enable_sharded_decode=True`` path on
``execute_chain_streaming``:
  - Construction validation (tokenizer required; tokenizer must
    expose .encode + .decode; cache_evict_send_message callable;
    sharded_default_max_tokens positive int)
  - Single-stage chain prefill + decode (smoke)
  - 2-stage chain produces autoregressive output (token feeds
    back into Stage 1 on each INCREMENTAL iteration)
  - ``max_tokens`` cap honored end-to-end (loop stops at the cap
    even when the tail never emits is_terminal=True)
  - EOS-reached → loop terminates (tail returns is_terminal=True
    mid-stream; remaining sequence_idx capacity unused)
  - Cancellation triggers eviction broadcast (caller closes the
    generator → cache_evict_send_message called for each stage)
  - Sampling params from request reach EVERY unary wire request
    (Phase 3.x.11 sharded contract: every chain stage's response
    can be a tail sample, so include_sampling_fields=True is
    threaded through unconditionally)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional, Tuple

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
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    encode_message,
    parse_message,
    response_input_commitment_for_request,
)
from prsm.compute.inference.multi_stage_attestation import (
    decode_multi_iteration_attestation,
    decode_multi_stage_attestation,
    is_multi_iteration_attestation,
    is_multi_stage_attestation,
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
# Test fakes
# ──────────────────────────────────────────────────────────────────────────


class _FakeAnchor:
    def __init__(self) -> None:
        self.registered: Dict[str, str] = {}

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


class _FakeTokenizer:
    """Minimal HF-AutoTokenizer-shaped stub.

    encode: maps str → list of int (length = number of words).
    decode: maps list of int → joined whitespace string with
    deterministic per-token text.
    """

    def __init__(self) -> None:
        self.encode_calls: List[str] = []
        self.decode_calls: List[List[int]] = []

    def encode(self, text: str) -> List[int]:
        self.encode_calls.append(text)
        # Each word → tok id = len(word). Empty string → [].
        words = text.split()
        return [max(1, len(w)) for w in words]

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        self.decode_calls.append(list(ids))
        # Each id → "tok{id}".
        return " ".join(f"tok{tid}" for tid in ids)


class _ShardedStageSim:
    """Per-stage simulator for sharded-decode.

    For non-tail stages: echoes input as hidden-state output.
    For the tail stage: pulls the next token from a configurable
    script; populates ``next_token_id`` + ``is_terminal``.

    Records every parsed request so tests can assert on
    ``decode_mode``, ``max_tokens``, ``temperature``, etc.
    """

    def __init__(
        self,
        identity,
        *,
        is_tail: bool,
        sample_script: Optional[List[Tuple[int, bool]]] = None,
        tee_type: TEEType = TEEType.SOFTWARE,
    ) -> None:
        self.identity = identity
        self.is_tail = is_tail
        self.tee_type = tee_type
        self.tee_attestation = b"\x02" * 32
        self.duration = 0.01
        self.epsilon = 0.0
        self._script: List[Tuple[int, bool]] = list(sample_script or [])
        self._cursor = 0
        self.calls: List[bytes] = []
        self.requests: List[RunLayerSliceRequest] = []

    def handle(self, request_bytes: bytes) -> bytes:
        self.calls.append(request_bytes)
        request = parse_message(request_bytes)
        assert isinstance(request, RunLayerSliceRequest)
        self.requests.append(request)

        # Echo the input as hidden state. For Stage 1 PREFILL with
        # int64 input_ids, transform to a float "hidden state" of
        # the same shape — the test harness only cares about
        # threading mechanics, not real model semantics.
        in_arr = decode_activation(
            request.activation_blob,
            request.activation_shape,
            request.activation_dtype,
        )
        if in_arr.dtype == np.int64:
            # Simulate token-embedding projection: int64 → float32
            # of same shape.
            out_arr = in_arr.astype(np.float32)
        else:
            out_arr = in_arr.astype(np.float32)
        out_blob, out_shape, out_dtype = encode_activation(out_arr)

        next_token_id: Optional[int] = None
        is_terminal = False
        if self.is_tail:
            if self._cursor >= len(self._script):
                raise AssertionError(
                    f"_ShardedStageSim: tail-sample script exhausted "
                    f"(cursor={self._cursor}, script_len="
                    f"{len(self._script)}; pad the script or shorten "
                    f"max_tokens)"
                )
            next_token_id, is_terminal = self._script[self._cursor]
            self._cursor += 1

        response = RunLayerSliceResponse.sign(
            identity=self.identity,
            request_id=request.request_id,
            activation_blob=out_blob,
            activation_shape=out_shape,
            activation_dtype=out_dtype,
            duration_seconds=self.duration,
            tee_attestation=self.tee_attestation,
            tee_type=self.tee_type,
            epsilon_spent=self.epsilon,
            next_token_id=next_token_id,
            is_terminal=is_terminal,
            # sp950 — mirror the real LayerStageServer's per-stage input-
            # commitment binding so the executor's corrected anchor verification
            # of non-PREFILL stages succeeds (pre-sp950 the sim signed without
            # it, matching the buggy executor-expects-None contract).
            input_commitment=response_input_commitment_for_request(request),
        )
        return encode_message(response)


class _ShardedTransport:
    def __init__(self, sims: Dict[str, _ShardedStageSim]) -> None:
        self.sims = sims
        self.delivery_log: List[str] = []

    def send(self, address: str, request_bytes: bytes) -> bytes:
        self.delivery_log.append(address)
        sim = self.sims.get(address)
        if sim is None:
            raise ConnectionError(f"no sharded sim at {address!r}")
        return sim.handle(request_bytes)


class _EvictionLog:
    """Records every broadcast call. Returns a parsed
    ``EvictCacheResponse(evicted=True)`` ack so the executor's
    ack-parsing path is exercised."""

    def __init__(self, *, evicted: bool = True) -> None:
        self.calls: List[Tuple[str, bytes]] = []
        self._evicted = evicted

    def __call__(self, address: str, payload: bytes) -> bytes:
        self.calls.append((address, payload))
        # Parse the broadcast envelope so the test can verify
        # the executor sent a real ``EvictCacheRequest``.
        msg = parse_message(payload)
        assert isinstance(msg, EvictCacheRequest)
        return encode_message(
            EvictCacheResponse(
                request_id=msg.request_id, evicted=self._evicted,
            )
        )


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _prompt_encoder_passthrough(prompt: str) -> np.ndarray:
    raw = prompt.encode("utf-8")
    pad = (4 - len(raw) % 4) % 4
    return np.frombuffer(raw + b"\x00" * pad, dtype=np.int32).copy()


def _output_decoder_passthrough(arr: np.ndarray) -> str:
    return arr.tobytes().rstrip(b"\x00").decode("utf-8", errors="replace")


def _make_chain(
    stages: List[str], total_layers: int = 4,
) -> GPUChain:
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
    prompt: str = "hello world",
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        model_id="test-model",
        budget_ftns=Decimal("10.0"),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        request_id="req-1",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _make_executor(
    *,
    transport: _ShardedTransport,
    settler,
    anchor,
    tokenizer: Optional[_FakeTokenizer] = None,
    enable_sharded_decode: bool = True,
    cache_evict_send_message=None,
    sharded_default_max_tokens: int = 16,
) -> RpcChainExecutor:
    return RpcChainExecutor(
        settler_identity=settler,
        send_message=transport.send,
        anchor=anchor,
        prompt_encoder=_prompt_encoder_passthrough,
        output_decoder=_output_decoder_passthrough,
        enable_sharded_decode=enable_sharded_decode,
        tokenizer=tokenizer,
        cache_evict_send_message=cache_evict_send_message,
        sharded_default_max_tokens=sharded_default_max_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────
# Construction validation
# ──────────────────────────────────────────────────────────────────────────


class TestShardedConstruction:
    def test_enable_sharded_decode_requires_tokenizer(self):
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()
        with pytest.raises(RuntimeError, match="tokenizer"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_encoder_passthrough,
                output_decoder=_output_decoder_passthrough,
                enable_sharded_decode=True,
                tokenizer=None,
            )

    def test_tokenizer_missing_encode_rejected(self):
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()

        class _Bad:
            def decode(self, ids):  # noqa: ARG002
                return ""
        with pytest.raises(RuntimeError, match="encode"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_encoder_passthrough,
                output_decoder=_output_decoder_passthrough,
                enable_sharded_decode=True,
                tokenizer=_Bad(),
            )

    def test_tokenizer_missing_decode_rejected(self):
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()

        class _Bad:
            def encode(self, text):  # noqa: ARG002
                return []
        with pytest.raises(RuntimeError, match="decode"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_encoder_passthrough,
                output_decoder=_output_decoder_passthrough,
                enable_sharded_decode=True,
                tokenizer=_Bad(),
            )

    def test_non_callable_cache_evict_rejected(self):
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()
        with pytest.raises(RuntimeError, match="cache_evict_send_message"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_encoder_passthrough,
                output_decoder=_output_decoder_passthrough,
                cache_evict_send_message="not callable",  # type: ignore[arg-type]
            )

    def test_invalid_sharded_default_max_tokens_rejected(self):
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()
        with pytest.raises(ValueError, match="sharded_default_max_tokens"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_encoder_passthrough,
                output_decoder=_output_decoder_passthrough,
                sharded_default_max_tokens=0,
            )

    def test_bool_sharded_default_max_tokens_rejected(self):
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()
        with pytest.raises(ValueError, match="sharded_default_max_tokens"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=lambda a, b: b"",
                anchor=anchor,
                prompt_encoder=_prompt_encoder_passthrough,
                output_decoder=_output_decoder_passthrough,
                sharded_default_max_tokens=True,  # type: ignore[arg-type]
            )

    def test_disabled_sharded_decode_does_not_require_tokenizer(self):
        # Default enable_sharded_decode=False: tokenizer can be None.
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()
        executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=lambda a, b: b"",
            anchor=anchor,
            prompt_encoder=_prompt_encoder_passthrough,
            output_decoder=_output_decoder_passthrough,
        )
        assert executor is not None


# ──────────────────────────────────────────────────────────────────────────
# Single-stage smoke
# ──────────────────────────────────────────────────────────────────────────


class TestShardedSingleStage:
    def test_single_stage_prefill_and_one_decode_smoke(self):
        # 1-stage chain — the only stage is the tail. PREFILL
        # produces token #1 (id=10); INCREMENTAL #1 produces
        # token #2 (id=20, is_terminal=True).
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64

        sim = _ShardedStageSim(
            alice, is_tail=True,
            sample_script=[(10, False), (20, True)],
        )
        transport = _ShardedTransport({alice.node_id: sim})
        tokenizer = _FakeTokenizer()
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=tokenizer,
        )
        chain = _make_chain([alice.node_id])

        events = list(executor.execute_chain_streaming(
            request=_make_request("hello", max_tokens=5),
            chain=chain,
        ))
        # Two StreamTokens then ChainExecutionResult.
        assert len(events) == 3
        assert isinstance(events[0], StreamToken)
        assert isinstance(events[1], StreamToken)
        assert isinstance(events[2], ChainExecutionResult)
        assert events[0].token_id == 10
        assert events[1].token_id == 20
        # Final result joins the two text deltas.
        assert events[2].output == events[0].text_delta + events[1].text_delta
        # Two unary requests sent (PREFILL + 1 INCREMENTAL).
        assert len(sim.requests) == 2
        assert sim.requests[0].decode_mode == DecodeMode.PREFILL
        assert sim.requests[1].decode_mode == DecodeMode.INCREMENTAL

    def test_decode_mode_threads_through_chain(self):
        # PREFILL sets decode_mode=PREFILL on the wire; INCREMENTAL
        # sets decode_mode=INCREMENTAL. Verifies the wire-format
        # threading.
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        sim = _ShardedStageSim(
            alice, is_tail=True,
            sample_script=[(10, False), (20, False), (30, True)],
        )
        transport = _ShardedTransport({alice.node_id: sim})
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
        )
        chain = _make_chain([alice.node_id])
        list(executor.execute_chain_streaming(
            request=_make_request("hi", max_tokens=10), chain=chain,
        ))
        assert [r.decode_mode for r in sim.requests] == [
            DecodeMode.PREFILL,
            DecodeMode.INCREMENTAL,
            DecodeMode.INCREMENTAL,
        ]


# ──────────────────────────────────────────────────────────────────────────
# Two-stage chain — autoregressive output
# ──────────────────────────────────────────────────────────────────────────


class TestShardedTwoStage:
    def test_two_stage_autoregressive_output(self):
        # 2-stage chain. Stage 1 (alice) is non-tail; Stage 2 (bob)
        # is the tail. Each iteration walks alice → bob; bob
        # samples next_token_id; loop feeds that into alice on the
        # next INCREMENTAL.
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        bob = generate_node_identity("bob")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        anchor.registered[bob.node_id] = bob.public_key_b64

        alice_sim = _ShardedStageSim(alice, is_tail=False)
        bob_sim = _ShardedStageSim(
            bob, is_tail=True,
            sample_script=[
                (100, False),  # token #1
                (200, False),  # token #2
                (300, True),   # token #3 — terminal
            ],
        )
        transport = _ShardedTransport({
            alice.node_id: alice_sim,
            bob.node_id: bob_sim,
        })
        tokenizer = _FakeTokenizer()
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=tokenizer,
        )
        chain = _make_chain(
            [alice.node_id, bob.node_id], total_layers=4,
        )

        events = list(executor.execute_chain_streaming(
            request=_make_request("hello world", max_tokens=10),
            chain=chain,
        ))
        # 3 StreamTokens + ChainExecutionResult.
        stream_tokens = [e for e in events if isinstance(e, StreamToken)]
        results = [e for e in events if isinstance(e, ChainExecutionResult)]
        assert len(stream_tokens) == 3
        assert len(results) == 1
        assert [t.token_id for t in stream_tokens] == [100, 200, 300]

        # Each iteration: 1 alice call + 1 bob call. 3 iterations
        # = 3 alice calls + 3 bob calls.
        assert len(alice_sim.requests) == 3
        assert len(bob_sim.requests) == 3
        # Token feedback: INCREMENTAL #1 → alice receives [100];
        # INCREMENTAL #2 → alice receives [200].
        # Stage 1 PREFILL receives the encoded prompt's input_ids
        # (FakeTokenizer emits [5, 5] for "hello world" since
        # both words are length 5).
        prefill_in = decode_activation(
            alice_sim.requests[0].activation_blob,
            alice_sim.requests[0].activation_shape,
            alice_sim.requests[0].activation_dtype,
        )
        assert prefill_in.tolist() == [5, 5]
        inc_in_1 = decode_activation(
            alice_sim.requests[1].activation_blob,
            alice_sim.requests[1].activation_shape,
            alice_sim.requests[1].activation_dtype,
        )
        assert inc_in_1.tolist() == [100]
        inc_in_2 = decode_activation(
            alice_sim.requests[2].activation_blob,
            alice_sim.requests[2].activation_shape,
            alice_sim.requests[2].activation_dtype,
        )
        assert inc_in_2.tolist() == [200]


# ──────────────────────────────────────────────────────────────────────────
# max_tokens cap
# ──────────────────────────────────────────────────────────────────────────


class TestShardedMaxTokens:
    def test_max_tokens_cap_honored_when_tail_never_terminates(self):
        # Tail returns is_terminal=False forever; loop must stop
        # at request.max_tokens.
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64

        # Pad script far past max_tokens — loop must stop short.
        script = [(i, False) for i in range(100)]
        sim = _ShardedStageSim(alice, is_tail=True, sample_script=script)
        transport = _ShardedTransport({alice.node_id: sim})
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
        )
        chain = _make_chain([alice.node_id])

        events = list(executor.execute_chain_streaming(
            request=_make_request("p", max_tokens=4), chain=chain,
        ))
        stream_tokens = [e for e in events if isinstance(e, StreamToken)]
        # Exactly 4 tokens — PREFILL produces #1, INC produces 2/3/4.
        assert len(stream_tokens) == 4

    def test_max_tokens_default_when_unset(self):
        # request.max_tokens=None → executor's
        # sharded_default_max_tokens is used.
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64

        script = [(i, False) for i in range(100)]
        sim = _ShardedStageSim(alice, is_tail=True, sample_script=script)
        transport = _ShardedTransport({alice.node_id: sim})
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
            sharded_default_max_tokens=3,
        )
        chain = _make_chain([alice.node_id])

        events = list(executor.execute_chain_streaming(
            request=_make_request("p"), chain=chain,
        ))
        stream_tokens = [e for e in events if isinstance(e, StreamToken)]
        # Default cap=3.
        assert len(stream_tokens) == 3


# ──────────────────────────────────────────────────────────────────────────
# EOS termination
# ──────────────────────────────────────────────────────────────────────────


class TestShardedEOS:
    def test_eos_terminates_loop_before_max_tokens(self):
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        # Tail emits 2 tokens then EOS on token #3.
        sim = _ShardedStageSim(
            alice, is_tail=True,
            sample_script=[(7, False), (8, False), (9, True)],
        )
        transport = _ShardedTransport({alice.node_id: sim})
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
        )
        chain = _make_chain([alice.node_id])
        events = list(executor.execute_chain_streaming(
            request=_make_request("p", max_tokens=100),  # huge cap
            chain=chain,
        ))
        stream_tokens = [e for e in events if isinstance(e, StreamToken)]
        assert len(stream_tokens) == 3
        # Last token's finish_reason is "stop".
        assert stream_tokens[-1].finish_reason == "stop"


# ──────────────────────────────────────────────────────────────────────────
# Cancellation eviction broadcast
# ──────────────────────────────────────────────────────────────────────────


class TestShardedCancellation:
    def test_caller_close_triggers_eviction_broadcast_to_all_stages(self):
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        bob = generate_node_identity("bob")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        anchor.registered[bob.node_id] = bob.public_key_b64

        alice_sim = _ShardedStageSim(alice, is_tail=False)
        bob_sim = _ShardedStageSim(
            bob, is_tail=True,
            sample_script=[(i, False) for i in range(100)],
        )
        transport = _ShardedTransport({
            alice.node_id: alice_sim,
            bob.node_id: bob_sim,
        })
        evict = _EvictionLog()
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
            cache_evict_send_message=evict,
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        gen = executor.execute_chain_streaming(
            request=_make_request("p", max_tokens=100), chain=chain,
        )
        # Pull two tokens then close.
        next(gen)
        next(gen)
        gen.close()
        # Eviction broadcast: both stages got an evict call.
        addresses = sorted(call[0] for call in evict.calls)
        assert addresses == sorted([alice.node_id, bob.node_id])
        for _, payload in evict.calls:
            msg = parse_message(payload)
            assert isinstance(msg, EvictCacheRequest)
            assert msg.request_id == "req-1"

    def test_terminal_completion_also_triggers_eviction_broadcast(self):
        # Eviction is on EVERY exit path — natural completion
        # also triggers broadcast.
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        sim = _ShardedStageSim(
            alice, is_tail=True,
            sample_script=[(1, True)],
        )
        transport = _ShardedTransport({alice.node_id: sim})
        evict = _EvictionLog()
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
            cache_evict_send_message=evict,
        )
        chain = _make_chain([alice.node_id])
        list(executor.execute_chain_streaming(
            request=_make_request("p", max_tokens=5), chain=chain,
        ))
        assert len(evict.calls) == 1
        addr, payload = evict.calls[0]
        assert addr == alice.node_id
        msg = parse_message(payload)
        assert isinstance(msg, EvictCacheRequest)
        assert msg.request_id == "req-1"

    def test_no_eviction_wire_no_op_no_raise(self):
        # cache_evict_send_message=None → eviction is a silent
        # no-op. The main flow must still complete cleanly.
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        sim = _ShardedStageSim(
            alice, is_tail=True,
            sample_script=[(1, True)],
        )
        transport = _ShardedTransport({alice.node_id: sim})
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
            cache_evict_send_message=None,
        )
        chain = _make_chain([alice.node_id])
        events = list(executor.execute_chain_streaming(
            request=_make_request("p", max_tokens=5), chain=chain,
        ))
        # Completed cleanly — final ChainExecutionResult yielded.
        assert any(
            isinstance(e, ChainExecutionResult) for e in events
        )

    def test_eviction_callable_raise_does_not_fail_main_flow(self):
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        sim = _ShardedStageSim(
            alice, is_tail=True,
            sample_script=[(1, True)],
        )
        transport = _ShardedTransport({alice.node_id: sim})

        def _raising_evict(addr: str, payload: bytes) -> bytes:
            raise ConnectionError("eviction transport down")

        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
            cache_evict_send_message=_raising_evict,
        )
        chain = _make_chain([alice.node_id])
        events = list(executor.execute_chain_streaming(
            request=_make_request("p", max_tokens=5), chain=chain,
        ))
        # Main flow completed despite eviction transport failure.
        assert any(
            isinstance(e, ChainExecutionResult) for e in events
        )


# ──────────────────────────────────────────────────────────────────────────
# Sampling params reach the wire
# ──────────────────────────────────────────────────────────────────────────


class TestShardedSamplingPropagation:
    def test_sampling_params_reach_every_unary_request(self):
        # Sharded contract: every unary RunLayerSliceRequest carries
        # the request's max_tokens + temperature (the tail reads
        # them at sample-time; non-tail stages ignore them but
        # the wire format still includes them so the tail's
        # signature can commit to the same envelope shape).
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        bob = generate_node_identity("bob")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        anchor.registered[bob.node_id] = bob.public_key_b64

        alice_sim = _ShardedStageSim(alice, is_tail=False)
        bob_sim = _ShardedStageSim(
            bob, is_tail=True,
            sample_script=[(1, True)],
        )
        transport = _ShardedTransport({
            alice.node_id: alice_sim,
            bob.node_id: bob_sim,
        })
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        list(executor.execute_chain_streaming(
            request=_make_request(
                "p", max_tokens=4, temperature=0.0,
            ),
            chain=chain,
        ))
        for req in alice_sim.requests + bob_sim.requests:
            assert req.max_tokens == 4
            assert req.temperature == 0.0

    def test_temperature_zero_propagates_for_greedy(self):
        # 0.0 is a real value — must not collapse to None.
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        sim = _ShardedStageSim(
            alice, is_tail=True,
            sample_script=[(1, True)],
        )
        transport = _ShardedTransport({alice.node_id: sim})
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
        )
        chain = _make_chain([alice.node_id])
        list(executor.execute_chain_streaming(
            request=_make_request("p", temperature=0.0, max_tokens=5),
            chain=chain,
        ))
        assert sim.requests[0].temperature == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.x Task 2 — per-iteration receipt attestation envelope
# ──────────────────────────────────────────────────────────────────────────


class TestShardedPerIterationAttestation:
    def _build_two_stage_with_eos_after_n(self, n_tokens: int):
        """Helper: 2-stage chain whose tail emits n_tokens then
        is_terminal=True. Returns (executor, alice_id, bob_id,
        alice_sim, bob_sim)."""
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        bob = generate_node_identity("bob")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        anchor.registered[bob.node_id] = bob.public_key_b64
        alice_sim = _ShardedStageSim(alice, is_tail=False)
        script = [(i + 100, i == n_tokens - 1) for i in range(n_tokens)]
        bob_sim = _ShardedStageSim(bob, is_tail=True, sample_script=script)
        transport = _ShardedTransport({
            alice.node_id: alice_sim,
            bob.node_id: bob_sim,
        })
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
        )
        return executor, alice, bob, alice_sim, bob_sim

    def test_one_iteration_receipt_uses_iteration_envelope(self):
        # PREFILL produces token #1 with is_terminal=True →
        # exactly 1 iteration. Receipt's tee_attestation MUST be
        # the multi-iteration envelope, not the legacy
        # multi-stage envelope.
        executor, alice, bob, _, _ = (
            self._build_two_stage_with_eos_after_n(1)
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        events = list(executor.execute_chain_streaming(
            request=_make_request("hi", max_tokens=5), chain=chain,
        ))
        result = next(
            e for e in events if isinstance(e, ChainExecutionResult)
        )
        # New envelope, not the legacy one.
        assert is_multi_iteration_attestation(result.tee_attestation)
        assert not is_multi_stage_attestation(result.tee_attestation)
        # Decode + assert iteration shape.
        iterations = decode_multi_iteration_attestation(
            result.tee_attestation,
        )
        assert iterations is not None
        assert len(iterations) == 1
        assert iterations[0].iteration_index == 0
        assert iterations[0].decode_mode == DecodeMode.PREFILL
        assert len(iterations[0].stage_records) == 2

    def test_four_iteration_receipt_has_one_record_per_iteration(self):
        # 4-token decode → 1 PREFILL + 3 INCREMENTALs = 4
        # iterations. Each iteration carries 2 stage records
        # (alice + bob).
        executor, alice, bob, _, _ = (
            self._build_two_stage_with_eos_after_n(4)
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        events = list(executor.execute_chain_streaming(
            request=_make_request("hi", max_tokens=10), chain=chain,
        ))
        result = next(
            e for e in events if isinstance(e, ChainExecutionResult)
        )
        iterations = decode_multi_iteration_attestation(
            result.tee_attestation,
        )
        assert iterations is not None
        assert len(iterations) == 4
        # iteration 0 = PREFILL; 1..3 = INCREMENTALs.
        assert iterations[0].decode_mode == DecodeMode.PREFILL
        for it in iterations[1:]:
            assert it.decode_mode == DecodeMode.INCREMENTAL
        # Each iteration has 2 stage records (alice + bob).
        for it in iterations:
            assert len(it.stage_records) == 2
            stage_node_ids = {s.stage_node_id for s in it.stage_records}
            assert stage_node_ids == {alice.node_id, bob.node_id}

    def test_iteration_envelope_aggregates_duration_and_epsilon(self):
        # The receipt's duration_seconds + epsilon_spent must
        # aggregate across ALL iterations, not just the last.
        executor, alice, bob, alice_sim, bob_sim = (
            self._build_two_stage_with_eos_after_n(3)
        )
        # Configure non-trivial per-stage duration.
        alice_sim.duration = 0.05
        bob_sim.duration = 0.07
        chain = _make_chain([alice.node_id, bob.node_id])
        events = list(executor.execute_chain_streaming(
            request=_make_request("hi", max_tokens=10), chain=chain,
        ))
        result = next(
            e for e in events if isinstance(e, ChainExecutionResult)
        )
        # 3 iterations × 2 stages × (0.05 + 0.07) = 0.36
        assert result.duration_seconds == pytest.approx(0.36)

    def test_non_sharded_execute_chain_receipt_unchanged(self):
        # Sanity: the unary execute_chain (non-sharded) still
        # produces the legacy multi-stage envelope. Phase 3.x.11.x
        # MUST NOT change non-sharded receipts. Pin via
        # is_multi_stage_attestation True + is_multi_iteration
        # False.
        from prsm.compute.chain_rpc.client import RpcChainExecutor
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        bob = generate_node_identity("bob")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        anchor.registered[bob.node_id] = bob.public_key_b64
        alice_sim = _ShardedStageSim(alice, is_tail=False)
        bob_sim = _ShardedStageSim(
            bob, is_tail=False,  # ignored by execute_chain
        )
        transport = _ShardedTransport({
            alice.node_id: alice_sim,
            bob.node_id: bob_sim,
        })
        executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=transport.send,
            anchor=anchor,
            prompt_encoder=_prompt_encoder_passthrough,
            output_decoder=_output_decoder_passthrough,
            enable_sharded_decode=False,  # NON-sharded
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(
            request=_make_request("hi", max_tokens=2),
            chain=chain,
        )
        # Legacy envelope is untouched.
        assert is_multi_stage_attestation(result.tee_attestation)
        assert not is_multi_iteration_attestation(result.tee_attestation)
        decoded = decode_multi_stage_attestation(result.tee_attestation)
        assert decoded is not None
        assert len(decoded) == 2

    def test_cancellation_mid_stream_no_receipt_emitted(self):
        # When the caller closes mid-stream, no
        # ChainExecutionResult is emitted (Phase 3.x.11 contract).
        # The eviction broadcast still fires via finally — see
        # TestShardedCancellation::test_caller_close_triggers_eviction_broadcast_to_all_stages.
        # This test confirms no partial receipt slips out, and
        # the new per-iteration envelope code-path doesn't change
        # that contract.
        executor, alice, bob, _, _ = (
            self._build_two_stage_with_eos_after_n(10)
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        gen = executor.execute_chain_streaming(
            request=_make_request("hi", max_tokens=10), chain=chain,
        )
        # Pull two tokens then close.
        tokens_seen = [next(gen), next(gen)]
        gen.close()
        assert all(isinstance(t, StreamToken) for t in tokens_seen)
        # No partial ChainExecutionResult ever yielded.
        # Generator's already closed; further next() raises
        # StopIteration; verifying the cancellation contract by
        # construction (above lines didn't see a result).


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.x Task 4 — executor lifts chunked + sharded PREFILL guard
# ──────────────────────────────────────────────────────────────────────────


class TestShardedChunkedPrefillExecutorGuard:
    """Phase 3.x.11.x Task 4: lift the executor-side guard for
    PREFILL when ``enable_sharded_decode=True`` AND the activation
    crosses ``chunk_threshold_bytes``. INCREMENTAL stays unary-only.

    Pre-3.x.11.x: ``_dispatch_stage`` raised ACTIVATION_TOO_LARGE
    for ANY chunked dispatch on a sharded executor (Phase 3.x.11
    Task 9 M1 belt-and-braces).
    Post-3.x.11.x: PREFILL chunked → routes through
    ``_dispatch_streamed`` → server's ``_dispatch_streamed_sharded``.
    INCREMENTAL chunked → still raises ACTIVATION_TOO_LARGE with
    the refined "single-position INCREMENTAL doesn't benefit
    from chunking" message.
    """

    def test_incremental_chunked_still_raises_activation_too_large(self):
        # The new guard fires on (enable_sharded_decode AND
        # decode_mode != PREFILL AND blob > chunk_threshold_bytes).
        # Isolate by calling _dispatch_stage directly with
        # synthesized inputs — bypasses the per-token chain loop
        # interleaving with PREFILL-streamed-not-wired paths.
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=lambda a, b: b"",
            anchor=anchor,
            prompt_encoder=_prompt_encoder_passthrough,
            output_decoder=_output_decoder_passthrough,
            enable_sharded_decode=True,
            tokenizer=_FakeTokenizer(),
            sharded_default_max_tokens=4,
            chunk_threshold_bytes=4,
        )
        # Stage 1 INCREMENTAL input: np.int64 single token = 8 bytes
        # (forces chunking under threshold=4).
        activation = np.array([42], dtype=np.int64)
        with pytest.raises(ChainExecutionError) as exc_info:
            executor._dispatch_stage(
                stage_index=0,
                stage_node_id=alice.node_id,
                layer_range=(0, 3),
                activation=activation,
                request=_make_request("hi", max_tokens=4),
                chain_total=1,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.INCREMENTAL,
                include_sampling_fields=True,
            )
        assert exc_info.value.code == ExecutorErrorCode.ACTIVATION_TOO_LARGE
        assert "INCREMENTAL is unary-only" in exc_info.value.message
        assert "single-position" in exc_info.value.message.lower()

    def test_non_sharded_chunked_path_unchanged(self):
        # Non-sharded executor + chunk_threshold_bytes=4 should
        # route through to _dispatch_streamed (regular runner)
        # WITHOUT firing the sharded INCREMENTAL guard. This pins
        # the back-compat invariant for Phase 3.x.7.1 callers.
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()
        executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=lambda a, b: b"",
            anchor=anchor,
            prompt_encoder=_prompt_encoder_passthrough,
            output_decoder=_output_decoder_passthrough,
            enable_sharded_decode=False,
            chunk_threshold_bytes=4,
            # No streamed_send_message wired → chunked dispatch
            # raises a DIFFERENT error (ACTIVATION_TOO_LARGE
            # because streamed transport not wired). This is the
            # PRE-3.x.11.x behavior; the new guard's INCREMENTAL
            # message must NOT appear.
        )
        # Construct a chain + request that would chunk.
        # Note: execute_chain (non-streaming) will hit the
        # streamed path because chunk_threshold_bytes=4 and
        # prompt_encoder produces > 4 bytes.
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request("hello world world", max_tokens=4),
                chain=_make_chain([generate_node_identity("alice").node_id]),
            )
        # The error code is ACTIVATION_TOO_LARGE (streamed transport
        # not wired) — but the MESSAGE must NOT contain the
        # INCREMENTAL-specific guard text. Pre-3.x.11.x
        # error message stays unchanged.
        assert exc_info.value.code == ExecutorErrorCode.ACTIVATION_TOO_LARGE
        assert "INCREMENTAL is unary-only" not in exc_info.value.message

    def test_non_chunked_sharded_path_unchanged(self):
        # Sharded executor + DEFAULT chunk_threshold_bytes (large
        # enough that small activations stay inline) → existing
        # Phase 3.x.11 inline path used. Sanity that the new
        # guard doesn't fire on the common case.
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        sim = _ShardedStageSim(
            alice, is_tail=True,
            sample_script=[(10, True)],
        )
        transport = _ShardedTransport({alice.node_id: sim})
        executor = _make_executor(
            transport=transport, settler=settler, anchor=anchor,
            tokenizer=_FakeTokenizer(),
        )
        chain = _make_chain([alice.node_id])
        # Should complete cleanly via inline path.
        events = list(executor.execute_chain_streaming(
            request=_make_request("hi", max_tokens=4),
            chain=chain,
        ))
        # No exceptions; reached final result.
        results = [
            e for e in events if isinstance(e, ChainExecutionResult)
        ]
        assert len(results) == 1

    def test_prefill_chunked_routes_to_streamed_transport(self):
        # The load-bearing positive case: chunked PREFILL on a
        # sharded executor lifts the M1 guard and routes to
        # streamed_send_message. We don't need a full E2E here;
        # the integration test in Task 5 covers the end-to-end
        # roundtrip. This unit test confirms the guard is
        # actually lifted (vs. the pre-3.x.11.x raise).
        settler = generate_node_identity("settler")
        alice = generate_node_identity("alice")
        anchor = _FakeAnchor()
        anchor.registered[alice.node_id] = alice.public_key_b64
        sim = _ShardedStageSim(
            alice, is_tail=True,
            sample_script=[(10, True)],
        )
        transport = _ShardedTransport({alice.node_id: sim})

        # Streamed-send observer: just records that the streamed
        # path was invoked + raises ACTIVATION_TOO_LARGE so the
        # test doesn't need to construct the full streamed-server
        # round-trip (Task 5 covers that).
        streamed_send_invocations = []

        def _streamed_send(addr, manifest_bytes, chunk_iter):
            # Drain chunk_iter to verify chunked path was reached.
            chunks = list(chunk_iter)
            streamed_send_invocations.append({
                "address": addr,
                "manifest_bytes": manifest_bytes,
                "chunk_count": len(chunks),
            })
            # Raise to short-circuit; Task 5 covers full round-trip.
            raise RuntimeError("test-shortcircuit")

        executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=transport.send,
            streamed_send_message=_streamed_send,
            anchor=anchor,
            prompt_encoder=_prompt_encoder_passthrough,
            output_decoder=_output_decoder_passthrough,
            enable_sharded_decode=True,
            tokenizer=_FakeTokenizer(),
            sharded_default_max_tokens=4,
            chunk_threshold_bytes=4,  # forces chunking
        )
        chain = _make_chain([alice.node_id])

        # Drain — expect a TRANSPORT_ERROR (test-shortcircuit
        # raised inside streamed_send) rather than
        # ACTIVATION_TOO_LARGE. This confirms PREFILL went
        # through the streamed path (M1 guard lifted).
        with pytest.raises(ChainExecutionError) as exc_info:
            list(executor.execute_chain_streaming(
                request=_make_request("hi", max_tokens=4),
                chain=chain,
            ))
        # NOT ACTIVATION_TOO_LARGE — guard was lifted.
        assert exc_info.value.code == ExecutorErrorCode.TRANSPORT_ERROR
        assert "test-shortcircuit" in exc_info.value.message
        # Streamed transport WAS invoked.
        assert len(streamed_send_invocations) == 1
