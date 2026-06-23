"""Phase 3.x.10 Task 1 — real autoregressive ``StreamingLayerRunner``.

Drop-in replacement for ``SyntheticStreamingRunner`` (Phase 3.x.8
Task 2). Implements the same ``StreamingLayerRunner`` Protocol —
no public-surface change downstream. Drives a HuggingFace-style
model's ``.generate()`` with a custom streamer adapter that
bridges the synchronous-callback emission into the Protocol's
generator-pull contract.

What this closes: the SyntheticStreamingRunner placeholder
caveats accumulated in MEMORY.md, audit-prep §7.4 + §7.5, and 5+
docstrings/commit messages. After this lands, "real
autoregressive decode" goes from caveat to fact — the streaming
wire path produces genuinely-distinct tokens from a real model,
not synthetic word splits.

What stays out of scope (carried forward):
  - **Sharded autoregressive** — each stage running once per
    token. The tail must host enough of the model to generate
    locally. Phase 3.x.11 lifts this.
  - **Stop sequences** — only EOS + ``max_tokens`` in v1.
  - **Constant-time padding for Tier C content** — timing
    side-channel exists; documented in the Phase 3.x.10
    threat-model memo. Phase 3.x.10.x.

Architecture: ``AutoregressiveStreamingRunner`` accepts a model
+ tokenizer at construction and a per-call ``prompt`` to drive
``.generate()``. The ``_HFStreamerAdapter`` mediates between HF's
``put(token_id)`` callback API and the Protocol's
``Iterator[StreamingChunk]`` pull API by buffering tokens during
the synchronous ``.generate()`` call and yielding them after.

v1 trade-off: the tokens are buffered before the first chunk is
yielded (because HF's streamer is sync-during-generate). Real-time
delivery happens at the SSE / MCP layer above — the user still
sees tokens as the chain produces them, just with a once-per-
generate buffering at the runner boundary. Async streaming during
generate (yielding interleaved with token production) is a Phase
3.x.10.x perf upgrade.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np

from prsm.compute.inference.streaming_runner import StreamingChunk
from prsm.compute.tee.models import PrivacyLevel, TEEType

# torch is an optional dependency at the runner boundary. Real-HF
# ``model.generate`` requires ``input_ids`` as a 2-d ``torch.Tensor``;
# tokenizer.encode returns ``List[int]``. The wrap step happens
# in-runner so production callers get HF-compat without each
# operator wiring it themselves. When torch isn't installed (test
# envs with mocked models), input_ids passes through untouched.
try:  # pragma: no cover — exercised by both branches in CI
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]


__all__ = [
    "AutoregressiveStreamingRunner",
    "SamplingDefaults",
]


# ──────────────────────────────────────────────────────────────────────────
# SamplingDefaults
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SamplingDefaults:
    """Per-runner sampling defaults. Request-level fields
    (``InferenceRequest.max_tokens``, ``.temperature``) override
    these on each call. ``temperature == 0`` triggers greedy
    decode (``do_sample=False``) regardless of ``top_k`` /
    ``top_p``.

    Defaults match common chat-model conventions; operators
    can tune per-runner if their model expects different sampling.
    """

    max_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95


# ──────────────────────────────────────────────────────────────────────────
# _HFStreamerAdapter — sync-callback → buffered-pull bridge
# ──────────────────────────────────────────────────────────────────────────


class _HFStreamerAdapter:
    """Wraps the HF ``TextStreamer`` callback contract.

    HuggingFace's ``model.generate(streamer=...)`` calls
    ``streamer.put(token_ids_tensor)`` once per generated token
    (or once per generated batch — we handle the ``[1, N]`` and
    ``[N]`` shapes the HF API uses). The adapter:

      1. Tracks raw integer token ids in ``self.token_ids``.
      2. Decodes them incrementally with ``skip_special_tokens=True``,
         using the standard HF ``TextStreamer`` buffer-and-flush
         pattern: a UTF-8 multi-byte character may span multiple
         tokens, so the adapter holds output until a whole-character
         boundary emerges (detected by absence of a trailing
         replacement-char ``"\\ufffd"``).
      3. Calls ``on_text(piece, token_id)`` with the next ready text
         piece + its triggering token id. Producer code (the
         runner) buffers these into ``StreamingChunk``s.

    The ``end()`` method is called by HF at the end of generation
    and flushes any tail buffer.

    Multi-byte handling tested in Task 2 covers emoji + CJK.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        on_text: Callable[[str, int], None],
        prompt_id_count: int = 0,
    ) -> None:
        self._tokenizer = tokenizer
        self._on_text = on_text
        # Accumulated token ids (used for incremental decode + EOS
        # detection by the runner via ``last_token_id`` property).
        self.token_ids: List[int] = []
        # Print position into the cumulative-decoded string. HF's
        # TextStreamer pattern: track how much has been emitted to
        # avoid re-emitting earlier text. New emissions are the
        # diff between the latest decode and ``_print_offset``.
        self._print_offset: int = 0
        # Phase 3.x.10.y Task 1 — HF prompt-echo fix.
        # Real-HF ``model.generate(streamer=...)`` for byte-level BPE
        # tokenizers (e.g., GPT-2 family) calls ``streamer.put()``
        # with the FULL ``input_ids + first_generated_token`` on the
        # first call. Without skip-prompt logic, our cumulative-
        # decode emits the prompt back as the first wire chunk
        # (observed in Phase 3.x.10.x's full-stack E2E for
        # distilgpt2). ``prompt_id_count`` tells the adapter how
        # many leading token ids belong to the prompt; the adapter
        # accumulates them for cumulative-decode position
        # correctness but advances ``_print_offset`` to the prompt's
        # decoded length the moment the boundary is crossed, so
        # subsequent ``_maybe_flush`` only emits text from
        # generated tokens onward. Mirrors HF's
        # ``TextStreamer(skip_prompt=True)`` semantics.
        # Default 0: no skipping (back-compat for callers that
        # don't set the count, e.g. test fakes).
        self._prompt_id_count: int = prompt_id_count
        # While ``_in_prompt_phase`` is True, ``put()`` accumulates
        # ids without flushing. Flips False the first time
        # ``len(token_ids) > prompt_id_count``; from that point on
        # ``put()`` flushes normally.
        self._in_prompt_phase: bool = prompt_id_count > 0

    @property
    def last_token_id(self) -> Optional[int]:
        return self.token_ids[-1] if self.token_ids else None

    def put(self, value: Any) -> None:
        # HF passes either a 0-d tensor, a 1-d ``[N]`` tensor, or a
        # 2-d ``[1, N]`` tensor. Normalize to a Python list of ints.
        # For byte-level BPE tokenizers, the FIRST put() call
        # carries the full input_ids + first generated token; the
        # ``prompt_id_count`` skip-prompt logic below filters that
        # back to just the generated portion before flushing.
        ids = _coerce_token_ids(value)
        if not ids:
            return
        for tid in ids:
            self.token_ids.append(int(tid))
        if self._in_prompt_phase:
            if len(self.token_ids) <= self._prompt_id_count:
                # Still in the prompt portion — don't flush yet.
                return
            # Just crossed the prompt boundary. Pin
            # ``_print_offset`` at the prompt's decoded length so
            # the upcoming ``_maybe_flush`` only emits text from
            # generated tokens onward.
            prompt_text = self._tokenizer.decode(
                self.token_ids[:self._prompt_id_count],
                skip_special_tokens=True,
            )
            self._print_offset = len(prompt_text)
            self._in_prompt_phase = False
        self._maybe_flush()

    def end(self) -> None:
        # Final flush at end-of-stream. The cumulative decode here
        # may include text that didn't get flushed during put()
        # because the buffer ended mid-character (e.g., the model
        # stopped between a 4-byte emoji's two BPE tokens).
        # End-of-stream may include a trailing replacement char if
        # generation truly ended mid-character — that's a
        # model/tokenizer issue, not ours; emit as-is.
        # Prompt-phase guard (Phase 3.x.10.y Task 1): if we never
        # crossed the prompt boundary, the model produced zero
        # generated tokens — nothing to emit (the prompt itself
        # MUST NOT leak as wire text).
        if self._in_prompt_phase:
            return
        text = self._cumulative_decode()
        if len(text) > self._print_offset:
            piece = text[self._print_offset:]
            tid = self.token_ids[-1] if self.token_ids else -1
            self._on_text(piece, tid)
            self._print_offset = len(text)

    def _maybe_flush(self) -> None:
        text = self._cumulative_decode()
        # Only flush if the cumulative decode ends on a whole-
        # character boundary. The replacement char ``"�"`` is
        # how HF's tokenizers signal an incomplete UTF-8 sequence.
        if text.endswith("�"):
            return
        if len(text) <= self._print_offset:
            # Nothing new (e.g., the new token contributed only
            # partial bytes to a multi-byte character; cumulative
            # output unchanged at the printable-text level).
            return
        piece = text[self._print_offset:]
        tid = self.token_ids[-1] if self.token_ids else -1
        self._on_text(piece, tid)
        self._print_offset = len(text)

    def _cumulative_decode(self) -> str:
        return self._tokenizer.decode(
            self.token_ids, skip_special_tokens=True,
        )


def _coerce_token_ids(value: Any) -> List[int]:
    """Normalize HF streamer ``put()`` arguments to a list of int
    token ids. HF passes a ``torch.Tensor`` (0-d / 1-d / 2-d),
    a numpy array, or a plain int. Tests inject lists directly.
    """
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(v) for v in value]
    # numpy / torch tensors expose ``.tolist()``; numpy further
    # collapses 0-d arrays to a scalar, so wrap defensively.
    if hasattr(value, "tolist"):
        as_list = value.tolist()
        if isinstance(as_list, int):
            return [as_list]
        if isinstance(as_list, list):
            # Could be nested ``[[1, 2, 3]]`` for a 2-d tensor.
            flat: List[int] = []
            for item in as_list:
                if isinstance(item, list):
                    flat.extend(int(x) for x in item)
                else:
                    flat.append(int(item))
            return flat
    raise TypeError(
        f"_HFStreamerAdapter.put: cannot coerce {type(value).__name__} "
        f"to token ids"
    )


# ──────────────────────────────────────────────────────────────────────────
# AutoregressiveStreamingRunner
# ──────────────────────────────────────────────────────────────────────────


class AutoregressiveStreamingRunner:
    """Real autoregressive decoder. Implements
    ``StreamingLayerRunner`` Protocol.

    Drop-in replacement for ``SyntheticStreamingRunner``: same
    Protocol, no public-surface change. The
    ``LayerStageServer.handle_token_stream`` path consumes
    instances of this class identically.

    Constructor args:
      model              The model handle. Must expose
                         ``.generate(input_ids=..., streamer=...,
                         max_new_tokens=..., temperature=...,
                         do_sample=..., top_k=..., top_p=...,
                         eos_token_id=...) -> token_ids_tensor``.
      tokenizer          HF AutoTokenizer-shaped object exposing
                         ``encode(text) -> List[int]``,
                         ``decode(ids, skip_special_tokens=True)
                         -> str``, and ``eos_token_id: int``.
      tee_attestation    Bytes the stage signs over (TEE-bound
                         identity). Same shape as
                         SyntheticStreamingRunner.
      tee_type           ``TEEType``.
      sampling_defaults  Default sampling params; request fields
                         override.
      prompt_provider    Callable: ``(layer_range, activation,
                         privacy_tier) -> str``. Returns the
                         prompt text to encode for this dispatch.
                         Production wires this to a server-side
                         registry keyed on ``request_id`` (the
                         executor stashes the prompt before
                         calling handle_token_stream); tests
                         inject deterministic fakes.

    Tail-only contract:
      ``run_layer_slice_streaming(is_final_stage=False)`` yields
      a single terminal ``StreamingChunk`` with
      ``finish_reason="error"`` (mapped server-side to
      INTERNAL_ERROR per Phase 3.x.8 Task 2). Sharded
      autoregressive decode is Phase 3.x.11.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        tee_attestation: bytes,
        tee_type: TEEType,
        sampling_defaults: Optional[SamplingDefaults] = None,
        prompt_provider: Callable[
            [Tuple[int, int], np.ndarray, PrivacyLevel], str
        ],
    ) -> None:
        if model is None or not hasattr(model, "generate"):
            raise RuntimeError(
                "AutoregressiveStreamingRunner requires a model with "
                ".generate(input_ids, streamer=, ...) — see HF "
                "transformers AutoModelForCausalLM"
            )
        if (
            tokenizer is None
            or not hasattr(tokenizer, "encode")
            or not hasattr(tokenizer, "decode")
        ):
            raise RuntimeError(
                "AutoregressiveStreamingRunner requires a tokenizer "
                "with .encode + .decode (HF AutoTokenizer-shaped)"
            )
        if not isinstance(tee_attestation, (bytes, bytearray)):
            raise RuntimeError(
                "AutoregressiveStreamingRunner requires tee_attestation "
                "as bytes"
            )
        if prompt_provider is None or not callable(prompt_provider):
            raise RuntimeError(
                "AutoregressiveStreamingRunner requires a callable "
                "prompt_provider(layer_range, activation, privacy_tier)"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._tee_attestation = bytes(tee_attestation)
        self._tee_type = tee_type
        self._defaults = sampling_defaults or SamplingDefaults()
        self._prompt_provider = prompt_provider

    # ── per-call sampling resolution ──────────────────────────────────

    def _effective_max_tokens(
        self, request: Any,
    ) -> int:
        """Resolve max_tokens from the request, falling back to
        the runner's default. ``request.max_tokens`` may be None
        (caller didn't specify) — fall back. If specified, use it
        directly (no min/max clamping; operators control the
        ceiling via the runner's default + the request validator
        upstream)."""
        rmax = getattr(request, "max_tokens", None)
        if rmax is None:
            return self._defaults.max_tokens
        return int(rmax)

    def _effective_temperature(
        self, request: Any,
    ) -> float:
        rtemp = getattr(request, "temperature", None)
        if rtemp is None:
            return self._defaults.temperature
        return float(rtemp)

    # ── streaming runner Protocol ─────────────────────────────────────

    def run_layer_slice_streaming(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
        request: Any = None,
    ) -> Iterator[StreamingChunk]:
        """Stream real autoregressive tokens.

        ``request`` is a Phase 3.x.10 extension to the Protocol —
        the runner needs ``request.max_tokens`` /
        ``.temperature`` to drive sampling. Phase 3.x.8's
        SyntheticStreamingRunner ignored the request because it
        decoded a pre-computed activation. Server-side Phase
        3.x.10.x will route the request through to the runner;
        v1 has the runner accept a fallback ``request=None`` path
        that uses runner defaults exclusively.
        """
        # Tail-only contract — sharded autoregressive deferred to
        # Phase 3.x.11. A non-tail dispatch yields exactly one
        # terminal chunk with finish_reason="error" so the
        # server's handle_token_stream maps to INTERNAL_ERROR
        # without preceding text frames.
        if not is_final_stage:
            yield StreamingChunk(
                sequence_index=0,
                text_delta="",
                finish_reason="error",
                full_output_text="",
                duration_seconds=0.0,
                tee_attestation=self._tee_attestation,
                tee_type=self._tee_type,
                epsilon_spent=0.0,
            )
            return

        prompt = self._prompt_provider(
            layer_range, activation, privacy_tier,
        )
        # Sprint 1231 — apply the model's chat template for instruct models on
        # the streaming path too (mirrors the unary encoder sp1230), so chat
        # models stream coherent output instead of completion-style. No-op for
        # base models (no chat_template) — byte-identical to the prior raw
        # tokenizer.encode(prompt). Gated by PRSM_PARALLAX_CHAT_TEMPLATE.
        from prsm.compute.inference.local_inference import (
            should_use_chat_template,
            parallax_chat_template_enabled,
            encode_ids_for_generation,
        )
        _use_ct = should_use_chat_template(
            self._tokenizer, enabled=parallax_chat_template_enabled(),
        )
        input_ids = encode_ids_for_generation(
            self._tokenizer, prompt, use_chat_template=_use_ct,
        )
        # Capture the prompt token count BEFORE the tensor wrap —
        # the adapter uses it to skip the prompt's text on the
        # first put() call (Phase 3.x.10.y Task 1 fix; HF's
        # TextStreamer-equivalent skip_prompt semantics for
        # byte-level BPE tokenizers).
        prompt_id_count = len(input_ids) if hasattr(
            input_ids, "__len__"
        ) else 0
        # Real-HF ``model.generate`` requires a 2-d ``[batch=1, seq_len]``
        # ``torch.Tensor``; tokenizer.encode returns ``List[int]``. Wrap
        # only when torch is available AND we got a list (already-tensor
        # shapes pass through). Test fakes that pass ``List[int]`` to a
        # mock generate() will get a tensor here too — the mock handles
        # it via duck-typed ``.tolist()``.
        if _torch is not None and isinstance(input_ids, list):
            input_ids = _torch.tensor(
                [input_ids], dtype=_torch.long,
            )

        # Sampling resolution.
        max_new_tokens = (
            self._effective_max_tokens(request) if request is not None
            else self._defaults.max_tokens
        )
        temperature = (
            self._effective_temperature(request) if request is not None
            else self._defaults.temperature
        )
        do_sample = temperature > 0
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)

        # Buffer for text pieces emitted by the streamer adapter.
        # HF generate() blocks during decode; the adapter's
        # on_text callback pushes pieces here synchronously. After
        # generate() returns, we yield them as StreamingChunks.
        pieces: List[Tuple[str, int]] = []

        def on_text(piece: str, tid: int) -> None:
            pieces.append((piece, tid))

        adapter = _HFStreamerAdapter(
            tokenizer=self._tokenizer,
            on_text=on_text,
            prompt_id_count=prompt_id_count,
        )

        start_ts = time.time()
        try:
            output_ids = self._model.generate(
                input_ids=input_ids,
                streamer=adapter,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=self._defaults.top_k,
                top_p=self._defaults.top_p,
                eos_token_id=eos_token_id,
            )
            # Defensive: HF normally calls adapter.end() inside
            # generate(); call here too in case the adapter never
            # got the signal (custom model, mocked test).
            adapter.end()
        except Exception as exc:  # noqa: BLE001
            # Mid-decode crash. The buffered ``pieces`` MUST be
            # emitted as non-terminal StreamingChunks first, then a
            # terminal error chunk with empty ``text_delta`` and
            # ``full_output_text=joined``. The server-side
            # joined-text invariant (``LayerStageServer.handle_token_stream``
            # asserts ``"".join(text_deltas) == terminal.full_output_text``)
            # would reject a single-chunk error path that put the
            # partial on ``full_output_text`` but not on the
            # delta sequence — receipt would never commit to the
            # partial output. Round-1 review H1 remediation.
            partial = "".join(p for p, _ in pieces)
            duration = time.time() - start_ts
            for i, (piece, tid) in enumerate(pieces):
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=piece,
                    token_id=tid if tid >= 0 else None,
                )
            yield StreamingChunk(
                sequence_index=len(pieces),
                text_delta="",
                finish_reason="error",
                full_output_text=partial,
                duration_seconds=duration,
                tee_attestation=self._tee_attestation,
                tee_type=self._tee_type,
                epsilon_spent=0.0,
            )
            return

        # Determine finish_reason from the raw token output.
        # HF's generate() output_ids is a tensor of [batch, total_len]
        # or [total_len]; coerce + read the last id.
        last_token_id = _last_token_id(output_ids)
        if (
            eos_token_id is not None
            and last_token_id == eos_token_id
        ):
            finish_reason = "stop"
        else:
            finish_reason = "max_tokens"

        # Compute joined output. ``adapter.end()`` may have
        # appended a tail piece, so use ``pieces`` after the
        # synchronous generate() returned.
        full_output = "".join(p for p, _ in pieces)
        duration = time.time() - start_ts

        # Edge case: no pieces emitted (e.g., model immediately
        # generated EOS). Yield a single terminal chunk with
        # empty text_delta.
        if not pieces:
            yield StreamingChunk(
                sequence_index=0,
                text_delta="",
                finish_reason=finish_reason,
                full_output_text="",
                duration_seconds=duration,
                tee_attestation=self._tee_attestation,
                tee_type=self._tee_type,
                epsilon_spent=0.0,
            )
            return

        # Yield non-terminal chunks.
        last_idx = len(pieces) - 1
        for i, (piece, tid) in enumerate(pieces):
            if i < last_idx:
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=piece,
                    token_id=tid if tid >= 0 else None,
                )
            else:
                # Terminal chunk: aggregate fields populated.
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=piece,
                    token_id=tid if tid >= 0 else None,
                    finish_reason=finish_reason,
                    full_output_text=full_output,
                    duration_seconds=duration,
                    tee_attestation=self._tee_attestation,
                    tee_type=self._tee_type,
                    epsilon_spent=0.0,
                )


def _last_token_id(output_ids: Any) -> Optional[int]:
    """Extract the last token id from HF generate()'s output.
    Handles tensor shape variations (1-d ``[total_len]`` vs 2-d
    ``[batch, total_len]``)."""
    if hasattr(output_ids, "tolist"):
        as_list = output_ids.tolist()
        if isinstance(as_list, list):
            if as_list and isinstance(as_list[0], list):
                # 2-d batched.
                return int(as_list[0][-1]) if as_list[0] else None
            # 1-d.
            return int(as_list[-1]) if as_list else None
        return int(as_list)
    if isinstance(output_ids, list) and output_ids:
        return int(output_ids[-1])
    return None


# ──────────────────────────────────────────────────────────────────────────
# EmbedderBackedStreamingRunner — sprint 693 F42 bypass
# ──────────────────────────────────────────────────────────────────────────


class EmbedderBackedStreamingRunner:
    """Sprint 693 F42 bypass — autoregressive streaming runner that
    works from the embedded activation directly.

    The standard ``AutoregressiveStreamingRunner`` requires a
    ``prompt_provider`` callable that returns the original prompt
    text so it can tokenize + generate. PRSM's pipeline already
    embedded the prompt (sprint 614/688 ``build_hf_prompt_encoder``
    runs ``wte(input_ids) + wpe(positions)``) and ships the
    activation tensor downstream — the original text isn't
    available server-side without a per-request_id registry (F42
    multi-sprint design problem).

    This runner sidesteps the registry entirely by using HF's
    ``model.generate(inputs_embeds=...)`` path: the activation IS
    the initial embedding fed into the transformer blocks, and HF
    handles the autoregressive loop (token sampling, position
    embedding for new tokens, re-embedding, block re-run with KV
    cache if supported by the model class).

    Tradeoffs:
      - Works for ANY causal LM that supports ``inputs_embeds=``
        on generate() (gpt2, llama, mistral, qwen, etc. — most do)
      - Token IDs of the prompt are unknown to the runner, so
        ``skip_prompt`` semantics don't apply — the streamer
        emits ONLY the generated continuation pieces (no prompt
        echo), which IS what we want for streaming output.
      - Same wire-frame format as AutoregressiveStreamingRunner —
        plug-in compatible with LayerStageServer's streaming_runner.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        tee_attestation: bytes,
        tee_type: TEEType,
        sampling_defaults: Optional[SamplingDefaults] = None,
    ) -> None:
        if model is None or not hasattr(model, "generate"):
            raise RuntimeError(
                "EmbedderBackedStreamingRunner requires a model "
                "with .generate(inputs_embeds=, streamer=, ...) — "
                "HF transformers AutoModelForCausalLM"
            )
        if (
            tokenizer is None
            or not hasattr(tokenizer, "decode")
        ):
            raise RuntimeError(
                "EmbedderBackedStreamingRunner requires a tokenizer "
                "with .decode (HF AutoTokenizer-shaped)"
            )
        if not isinstance(tee_attestation, (bytes, bytearray)):
            raise RuntimeError(
                "EmbedderBackedStreamingRunner requires "
                "tee_attestation as bytes"
            )
        self._model = model
        self._tokenizer = tokenizer
        self._tee_attestation = bytes(tee_attestation)
        self._tee_type = tee_type
        self._defaults = sampling_defaults or SamplingDefaults()

    def _effective_max_tokens(self, request: Any) -> int:
        rmax = getattr(request, "max_tokens", None) if request else None
        if rmax is None:
            return self._defaults.max_tokens
        return int(rmax)

    def _effective_temperature(self, request: Any) -> float:
        rtemp = getattr(request, "temperature", None) if request else None
        if rtemp is None:
            return self._defaults.temperature
        return float(rtemp)

    def run_layer_slice_streaming(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
        request: Any = None,
    ) -> Iterator[StreamingChunk]:
        # Tail-only contract (mirror AutoregressiveStreamingRunner)
        if not is_final_stage:
            yield StreamingChunk(
                sequence_index=0,
                text_delta="",
                finish_reason="error",
                full_output_text="",
                duration_seconds=0.0,
                tee_attestation=self._tee_attestation,
                tee_type=self._tee_type,
                epsilon_spent=0.0,
            )
            return

        if _torch is None:
            raise RuntimeError(
                "EmbedderBackedStreamingRunner requires torch installed"
            )

        # Convert numpy activation → torch tensor [B, S, H]
        embeds = _torch.from_numpy(activation).to(
            dtype=_torch.float32,
        )
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)

        max_new_tokens = self._effective_max_tokens(request)
        temperature = self._effective_temperature(request)
        do_sample = temperature > 0
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)

        pieces: List[Tuple[str, int]] = []

        def on_text(piece: str, tid: int) -> None:
            pieces.append((piece, tid))

        adapter = _HFStreamerAdapter(
            tokenizer=self._tokenizer,
            on_text=on_text,
            # inputs_embeds path has no prompt token ids to skip —
            # generate() yields ONLY continuation pieces directly.
            prompt_id_count=0,
        )

        start_ts = time.time()
        try:
            output_ids = self._model.generate(
                inputs_embeds=embeds,
                streamer=adapter,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=self._defaults.top_k,
                top_p=self._defaults.top_p,
                eos_token_id=eos_token_id,
            )
            adapter.end()
        except Exception as exc:  # noqa: BLE001
            partial = "".join(p for p, _ in pieces)
            duration = time.time() - start_ts
            for i, (piece, tid) in enumerate(pieces):
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=piece,
                    token_id=tid if tid >= 0 else None,
                )
            yield StreamingChunk(
                sequence_index=len(pieces),
                text_delta="",
                finish_reason="error",
                full_output_text=partial,
                duration_seconds=duration,
                tee_attestation=self._tee_attestation,
                tee_type=self._tee_type,
                epsilon_spent=0.0,
            )
            return

        last_token_id = _last_token_id(output_ids)
        if eos_token_id is not None and last_token_id == eos_token_id:
            finish_reason = "stop"
        else:
            finish_reason = "max_tokens"

        full_output = "".join(p for p, _ in pieces)
        duration = time.time() - start_ts

        if not pieces:
            yield StreamingChunk(
                sequence_index=0,
                text_delta="",
                finish_reason=finish_reason,
                full_output_text="",
                duration_seconds=duration,
                tee_attestation=self._tee_attestation,
                tee_type=self._tee_type,
                epsilon_spent=0.0,
            )
            return

        last_idx = len(pieces) - 1
        for i, (piece, tid) in enumerate(pieces):
            if i < last_idx:
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=piece,
                    token_id=tid if tid >= 0 else None,
                )
            else:
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=piece,
                    token_id=tid if tid >= 0 else None,
                    finish_reason=finish_reason,
                    full_output_text=full_output,
                    duration_seconds=duration,
                    tee_attestation=self._tee_attestation,
                    tee_type=self._tee_type,
                    epsilon_spent=0.0,
                )
