"""Sprint 1019 — real single-node inference (Tier 2: real model by default).

Assembles a RUNNABLE ``ParallaxScheduledExecutor`` whose chain executor runs a
REAL HuggingFace causal-LM (gpt2 / distilgpt2) IN-PROCESS via ``transformers``
``model.generate`` — no RPC, no multi-host swarm, no ~8-env-var operator
ceremony, no Base-anchor identity. Opt-in via
``PRSM_INFERENCE_EXECUTOR=local``.

This is the single-node, whole-model counterpart to the distributed
``AutoregressiveStreamingRunner`` (which runs layer slices over RPC). Real
greedy/deterministic tokens flow through the GENUINE executor path (gates,
trust-filtered single-GPU pool, Phase-1 allocation, the execute_chain /
execute_chain_streaming drive loops, signed-receipt construction).

Honest scope: the MODEL output is real, but the trust attestation is
software-tier (``TEEType.SOFTWARE``, ``vendor_verified=false``) — real hardware
TEE (Intel DCAP / AMD KDS) is Tier 3. Use the ``local`` executor for real-output
dev/dogfood and tier-A public models; it makes NO hardware-confidentiality
guarantee.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from prsm.compute.chain_rpc.client import StreamToken
from prsm.compute.inference.parallax_executor import (
    ChainExecutionResult,
    ParallaxScheduledExecutor,
)
from prsm.compute.parallax_scheduling.model_info import ModelInfo
from prsm.compute.parallax_scheduling.prsm_request_router import (
    InMemoryProfileSource,
    ProfileSnapshot,
)
from prsm.compute.parallax_scheduling.prsm_types import ParallaxGPU
from prsm.compute.parallax_scheduling.trust_adapter import (
    AnchorVerifyAdapter,
    ConsensusMismatchHook,
    StakeWeightedTrustAdapter,
    TierGateAdapter,
    TrustStack,
)
from prsm.compute.tee.models import TEEType

# Models this executor can serve locally. The dims are the gpt2 family's real
# config (distilgpt2 inherits gpt2's dims with fewer layers) — used only for the
# allocator's catalog; actual generation reads the loaded model. Both are tiny,
# public, and (in this repo's dev image) cached for offline use.
_GPT2_FAMILY_DIMS = dict(
    head_size=64, hidden_dim=768, intermediate_dim=3072,
    num_attention_heads=12, num_kv_heads=12, vocab_size=50257,
)
logger = logging.getLogger(__name__)

_KNOWN_MODELS: Dict[str, int] = {"gpt2": 12, "distilgpt2": 6}

DEFAULT_LOCAL_MODEL = "distilgpt2"  # fastest on CPU
_DEFAULT_MAX_TOKENS = 32
_MAX_TOKENS_CEILING = 256

# Software-tier attestation — honest scope (NOT a hardware TEE).
_SOFTWARE_ATTESTATION = b"local-hf-runner-software-attestation"

_OFFLINE_ENV = "PRSM_LOCAL_INFERENCE_OFFLINE"
_PREWARM_ENV = "PRSM_LOCAL_INFERENCE_PREWARM"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _prewarm_enabled(explicit: Optional[bool]) -> bool:
    """sp1302 — resolve whether to pre-warm the model at startup. Explicit arg wins;
    else PRSM_LOCAL_INFERENCE_PREWARM (default TRUE = load the model OFF the request
    path so the first /compute/inference doesn't pay the cold-load cost: ~350MB
    download + load on a fresh install). 0/off/false/no disables."""
    if explicit is not None:
        return bool(explicit)
    raw = (os.environ.get(_PREWARM_ENV, "") or "").strip().lower()
    return True if raw == "" else raw in _TRUTHY


def _resolve_offline(explicit: Optional[bool]) -> bool:
    """sp1184 — resolve the offline (local-files-only) flag. Explicit arg wins; else read
    PRSM_LOCAL_INFERENCE_OFFLINE (default FALSE = allow a one-time hub download so a fresh
    install's first request succeeds). True = airgapped/offline-only."""
    if explicit is not None:
        return bool(explicit)
    return (os.environ.get(_OFFLINE_ENV, "") or "").strip().lower() in _TRUTHY


def resolve_model_revision(model_id: str) -> Optional[str]:
    """sp1285 — pin a model's HuggingFace artifacts to an IMMUTABLE commit.

    PRSM_MODEL_REVISIONS is a JSON object {model_id: revision}; this returns the pinned
    revision (commit sha / tag) for ``model_id``, or None (the default branch — prior
    behavior). Pinning a revision is the supply-chain defense: from_pretrained otherwise
    fetches whatever the model repo's default branch currently points at, so a hijacked or
    retagged HF repo could serve poisoned weights to the node."""
    raw = (os.environ.get("PRSM_MODEL_REVISIONS", "") or "").strip()
    if not raw:
        return None
    try:
        import json
        mapping = json.loads(raw)
        rev = mapping.get(model_id) if isinstance(mapping, dict) else None
        return rev if isinstance(rev, str) and rev.strip() else None
    except (ValueError, TypeError):
        logger.warning("PRSM_MODEL_REVISIONS is not valid JSON — ignoring (no revision pin)")
        return None


# ── sp1204: real inference quality (GPU + instruct model, greedy/§7-deterministic) ────
_DEVICE_ENV = "PRSM_LOCAL_INFERENCE_DEVICE"      # cuda | mps | cpu | auto (default auto)
_DTYPE_ENV = "PRSM_LOCAL_INFERENCE_DTYPE"        # float16 | bfloat16 | float32 | auto
_CHAT_TEMPLATE_ENV = "PRSM_LOCAL_INFERENCE_CHAT_TEMPLATE"  # 0 disables (default on)
_PARALLAX_CHAT_TEMPLATE_ENV = "PRSM_PARALLAX_CHAT_TEMPLATE"  # 0 disables (default on); distributed path
_VALID_DEVICES = frozenset({"cuda", "mps", "cpu"})
_VALID_DTYPES = frozenset({"float16", "bfloat16", "float32"})


def resolve_device(explicit: Optional[str], *, cuda: bool, mps: bool) -> str:
    """Pick the torch device. Explicit env (PRSM_LOCAL_INFERENCE_DEVICE) wins when valid;
    else cuda > mps > cpu. So a Lambda A10 uses the GPU automatically, a Mac uses MPS, and
    a plain box falls back to CPU — no config needed."""
    e = (explicit or "").strip().lower()
    if e in _VALID_DEVICES:
        return e
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


def resolve_dtype(explicit: Optional[str], device: str) -> str:
    """Pick the load dtype. Explicit env wins; else float16 on CUDA (≈2x less VRAM →
    a 7-8B instruct model fits a 24GB A10), float32 elsewhere (CPU/MPS fp16 is slow or
    flaky). Greedy decoding stays deterministic at any dtype (§7-safe)."""
    e = (explicit or "").strip().lower()
    if e in _VALID_DTYPES:
        return e
    return "float16" if device == "cuda" else "float32"


def should_use_chat_template(tokenizer: Any, *, enabled: bool) -> bool:
    """True iff the model is an INSTRUCT model (its tokenizer ships a chat_template) and
    the operator hasn't disabled templating. Instruct models produce coherent output ONLY
    when the prompt is wrapped in their chat template; base models (gpt2) have none → raw
    encoding. This is the single biggest quality lever."""
    return bool(enabled and getattr(tokenizer, "chat_template", None))


def encode_for_generation(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> Any:
    """Build the model inputs. Instruct path: apply_chat_template([{user, prompt}],
    add_generation_prompt=True) so the model sees its native instruction framing. Base
    path: raw encode (unchanged pre-sp1204 behavior). Returns an enc mapping with at least
    ``input_ids`` (and usually ``attention_mask``).

    sp1208: transformers v5's apply_chat_template returns a BatchEncoding (a dict-like with
    input_ids + attention_mask), so the old ``{"input_ids": <result>}`` wrap put a
    BatchEncoding under input_ids and broke ``torch.ones_like``. Request
    ``return_dict=True`` (BatchEncoding on 4.41+/v5) and pass it through; on older
    transformers that don't accept return_dict, fall back to the bare-tensor form + wrap."""
    if use_chat_template:
        msgs = [{"role": "user", "content": prompt}]
        try:
            out = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        except TypeError:  # older transformers: no return_dict kwarg → bare tensor
            out = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt")
        if isinstance(out, dict) or hasattr(out, "keys"):
            return out  # BatchEncoding/dict: already has input_ids (+ attention_mask)
        return {"input_ids": out}  # bare tensor → wrap (caller adds attention_mask)
    return tokenizer(prompt, return_tensors="pt")


def _chat_template_enabled(env: Optional[dict] = None) -> bool:
    e = env if env is not None else os.environ
    return (e.get(_CHAT_TEMPLATE_ENV, "") or "").strip().lower() not in {"0", "false", "no", "off"}


def parallax_chat_template_enabled(env: Optional[dict] = None) -> bool:
    """Sprint 1230/1231 — single source of truth for the parallax (distributed)
    chat-template toggle. ``PRSM_PARALLAX_CHAT_TEMPLATE`` in {0,false,no,off}
    disables it (independent of the single-node ``PRSM_LOCAL_INFERENCE_CHAT_TEMPLATE``).
    Used by BOTH the unary encoder (chain_executor_adapters) and the streaming
    runner (autoregressive_runner)."""
    e = env if env is not None else os.environ
    return (e.get(_PARALLAX_CHAT_TEMPLATE_ENV, "") or "").strip().lower() not in {
        "0", "false", "no", "off",
    }


def encode_ids_for_generation(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> Any:
    """Sprint 1231 — like ``encode_for_generation`` but returns a flat
    ``List[int]`` of token ids (the shape the streaming/autoregressive path
    needs: ``len()`` for prompt-skip + a manual tensor wrap). Instruct path:
    ``apply_chat_template([{user, prompt}], add_generation_prompt=True)`` with
    NO ``return_tensors`` → HF returns a token-id list. Base path: raw
    ``tokenizer.encode(prompt)`` (unchanged behavior)."""
    if use_chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
        )
        # default (no return_tensors) yields List[int]; a tokenizer that hands
        # back a BatchEncoding/dict gets normalized to its input_ids list.
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return ids
    return tokenizer.encode(prompt)


def dtype_kwarg_for_transformers(version: str) -> str:
    """sp1206 — pick from_pretrained's dtype kwarg name. transformers v5 renamed
    ``torch_dtype`` → ``dtype`` (and removed ``torch_dtype``); 4.x uses ``torch_dtype``.
    Getting this wrong is not cosmetic: an ignored dtype kwarg loads a 7B model in
    float32 (~28GB) and OOMs a 24GB A10 — the load MUST honor float16."""
    try:
        major = int(str(version).split(".")[0])
    except (ValueError, IndexError, AttributeError):
        major = 4
    return "dtype" if major >= 5 else "torch_dtype"


class LocalHuggingFaceChainExecutor:
    """ChainExecutor that runs a real HF causal-LM in-process (single stage =
    whole model). Implements both ``execute_chain`` (unary, for /compute/inference)
    and ``execute_chain_streaming`` (for /compute/inference/stream). Greedy
    (deterministic) decode so output is reproducible. Lazy-loads the model on
    first use (construction stays cheap)."""

    def __init__(self, *, model_id: str = DEFAULT_LOCAL_MODEL,
                 max_tokens: int = _DEFAULT_MAX_TOKENS,
                 offline: Optional[bool] = None) -> None:
        self._model_id = model_id
        self._default_max_tokens = max(1, int(max_tokens))
        # sp1184 — offline resolves from the arg, else PRSM_LOCAL_INFERENCE_OFFLINE, else
        # FALSE (day-one default): the model is DOWNLOADED on first use if not cached, so a
        # fresh install's first /compute/inference actually succeeds instead of failing on
        # local_files_only with an empty HF cache. offline=True restores the
        # airgapped/deterministic no-network behavior.
        self._offline = _resolve_offline(offline)
        self._use_chat_template = _chat_template_enabled()
        self._device = "cpu"
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "local inference requires 'transformers' + 'torch' "
                "(pip install -e '.[ml]'). " + str(exc)
            ) from exc
        # sp1204 — auto-detect GPU + dtype so a Lambda A10 (or a Mac MPS) is used without
        # config; CPU/float32 fallback preserves the pre-sp1204 behavior on a plain box.
        mps = bool(getattr(getattr(torch, "backends", None), "mps", None)
                   and torch.backends.mps.is_available())
        device = resolve_device(os.environ.get(_DEVICE_ENV),
                                cuda=bool(torch.cuda.is_available()), mps=mps)
        dtype_str = resolve_dtype(os.environ.get(_DTYPE_ENV), device)
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype_str]
        # local_files_only=self._offline: False → allow a one-time hub download; True →
        # offline-only, deterministic, no network round-trip.
        # sp1285 — pin to an immutable HF commit when configured (supply-chain: don't trust
        # the model repo's mutable default branch). None = default branch (prior behavior).
        _revision = resolve_model_revision(self._model_id)
        tok = AutoTokenizer.from_pretrained(
            self._model_id, local_files_only=self._offline, revision=_revision)
        # sp1206 — use the dtype kwarg the installed transformers expects (v5 renamed
        # torch_dtype→dtype). An ignored dtype loads fp32 → a 7B OOMs a 24GB A10.
        import transformers as _tf
        _dtype_kw = dtype_kwarg_for_transformers(getattr(_tf, "__version__", "4"))
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id, local_files_only=self._offline, revision=_revision,
                **{_dtype_kw: torch_dtype})
        except TypeError:  # version-detect missed → try the other kwarg name
            _other = "torch_dtype" if _dtype_kw == "dtype" else "dtype"
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id, local_files_only=self._offline, revision=_revision,
                **{_other: torch_dtype})
        model.eval()
        model.to(device)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self._tokenizer = tok
        self._model = model
        self._device = device
        logger.info(
            "local inference loaded model=%s device=%s dtype=%s chat_template=%s",
            self._model_id, device, dtype_str,
            should_use_chat_template(tok, enabled=self._use_chat_template))

    @property
    def is_loaded(self) -> bool:
        """True once the model + tokenizer are resident (after warm()/first use)."""
        return self._model is not None

    def warm(self) -> bool:
        """sp1302 — idempotently load the model OFF the request path (startup
        pre-warm) so the first /compute/inference doesn't pay the cold-load cost
        (~350MB download + load on a fresh install). FAIL-SOFT: a load error (no
        network, missing [ml], bad model id) is logged and returns False — the lazy
        ``_ensure_loaded`` on the first real request is unchanged, so pre-warm can
        never break serving or crash startup."""
        if self._model is not None:
            return True
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:  # noqa: BLE001 — pre-warm must never crash startup
            logger.warning(
                "local inference pre-warm failed for model=%s (%s); the model will "
                "load lazily on the first request instead.", self._model_id, exc)
            return False

    def describe(self) -> Dict[str, Any]:
        """Readiness snapshot for status/health surfaces. Never triggers a load."""
        return {
            "model_id": self._model_id,
            "loaded": self._model is not None,
            "device": self._device if self._model is not None else None,
            "offline": self._offline,
            "max_tokens": self._default_max_tokens,
        }

    def _resolve_max_tokens(self, request: Any) -> int:
        mt = getattr(request, "max_tokens", None)
        if isinstance(mt, int) and mt > 0:
            return min(mt, _MAX_TOKENS_CEILING)
        return self._default_max_tokens

    def _generate(self, request: Any) -> Tuple[List[str], str, bool]:
        """Run greedy generation. Returns (per-token deltas, full new text,
        hit_length). The deltas concatenate to the full text (receipt-hash
        invariant); they are derived by cumulative decode so byte-level BPE
        tokens that split a character are handled."""
        import torch

        self._ensure_loaded()
        prompt = getattr(request, "prompt", "") or ""
        max_new = self._resolve_max_tokens(request)
        # sp1204 — wrap the prompt in the model's chat template when it's an instruct
        # model (coherent output vs gibberish); raw encode for base models (gpt2).
        use_ct = should_use_chat_template(self._tokenizer, enabled=self._use_chat_template)
        enc = encode_for_generation(self._tokenizer, prompt, use_chat_template=use_ct)
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        enc = {k: v.to(self._device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,  # greedy → deterministic, reproducible
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = out[0][prompt_len:].tolist()
        deltas: List[str] = []
        prev = ""
        for i in range(len(new_ids)):
            cur = self._tokenizer.decode(new_ids[: i + 1], skip_special_tokens=True)
            deltas.append(cur[len(prev):])
            prev = cur
        hit_length = len(new_ids) >= max_new
        return deltas, prev, hit_length

    def _terminal(self, output: str, duration: float) -> ChainExecutionResult:
        return ChainExecutionResult(
            output=output,
            duration_seconds=max(0.0, duration),
            tee_attestation=_SOFTWARE_ATTESTATION,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )

    def execute_chain(self, *, request: Any, chain: Any) -> ChainExecutionResult:  # noqa: ARG002
        t0 = time.monotonic()
        _deltas, full, _ = self._generate(request)
        return self._terminal(full, time.monotonic() - t0)

    def execute_chain_streaming(
        self, *, request: Any, chain: Any,  # noqa: ARG002
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        t0 = time.monotonic()
        deltas, full, hit_length = self._generate(request)
        last = len(deltas) - 1
        for i, delta in enumerate(deltas):
            yield StreamToken(
                sequence_index=i,
                text_delta=delta,
                finish_reason=("length" if hit_length else "stop") if i == last else None,
            )
        yield self._terminal(full, time.monotonic() - t0)


def _first_attr(config: Any, *names: str) -> Any:
    """Return the first present, non-None attribute among ``names`` (or None).

    PretrainedConfig families spell the same field differently (gpt2: ``n_layer``;
    modern Llama/Mistral/Qwen: ``num_hidden_layers``), so derivation must probe
    several aliases per logical field.
    """
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _derive_model_info(config: Any, model_id: str) -> ModelInfo:
    """Map a HF-config-like object (anything exposing the attributes) to a
    ``ModelInfo`` for the allocator catalog (which drives chain-stage layer
    slicing).

    Handles BOTH naming schemes via cross-aliased ``getattr``:
      * gpt2-style: ``n_layer`` / ``n_embd`` / ``n_head`` / ``n_inner`` / ``vocab_size``
        (gpt2 is MHA, so it carries no kv-head field).
      * modern (Llama/Mistral/Qwen): ``num_hidden_layers`` / ``hidden_size`` /
        ``num_attention_heads`` / ``num_key_value_heads`` / ``intermediate_size`` /
        ``vocab_size`` (GQA: kv-heads may be < attention-heads).

    Derivation rules:
      * num_kv_heads defaults to num_attention_heads when ``num_key_value_heads``
        is absent (pure MHA).
      * head_size uses an explicit ``head_dim`` when present, else
        hidden_dim // num_attention_heads (768 // 12 == 64 for gpt2).
      * intermediate_dim falls back to 4 * hidden_dim when ``n_inner`` /
        ``intermediate_size`` is absent/None (the standard transformer ratio).

    This is the pure, network-free core: ``_model_info_for`` is the thin loader
    that obtains ``config`` from ``AutoConfig`` (and falls back to the gpt2
    hardcode on any load failure).
    """
    _n_layers = _first_attr(config, "num_hidden_layers", "n_layer")
    num_layers = int(_n_layers) if _n_layers is not None else 12
    hidden_dim = int(_first_attr(config, "hidden_size", "n_embd"))
    num_attention_heads = int(_first_attr(config, "num_attention_heads", "n_head"))
    num_kv_heads = _first_attr(config, "num_key_value_heads", "num_kv_heads")
    num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else num_attention_heads

    explicit_head_dim = _first_attr(config, "head_dim", "head_size")
    head_size = (
        int(explicit_head_dim)
        if explicit_head_dim is not None
        else hidden_dim // num_attention_heads
    )

    intermediate = _first_attr(config, "intermediate_size", "n_inner")
    intermediate_dim = int(intermediate) if intermediate is not None else 4 * hidden_dim

    vocab_size = int(_first_attr(config, "vocab_size"))

    return ModelInfo(
        model_name=model_id,
        mlx_model_name=model_id + "-mlx",
        head_size=head_size,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )


def _gpt2_hardcode_model_info(model_id: str) -> ModelInfo:
    """The proven, in-production gpt2-family ModelInfo — the resilient fallback
    when ``AutoConfig`` cannot load (offline / uncached / no ``transformers``)."""
    num_layers = _KNOWN_MODELS.get(model_id, 12)
    return ModelInfo(
        model_name=model_id,
        mlx_model_name=model_id + "-mlx",
        num_layers=num_layers,
        **_GPT2_FAMILY_DIMS,
    )


def _model_info_for(model_id: str) -> ModelInfo:
    """Build the allocator-catalog ``ModelInfo`` for ``model_id`` from its real
    HF config so layer slicing is correct for ANY model (not just gpt2).

    Tries ``AutoConfig.from_pretrained(model_id, local_files_only=True)`` and
    delegates to ``_derive_model_info``. On ANY load failure (offline, uncached,
    missing ``transformers``, or a config missing a required field) it falls back
    to the proven gpt2 hardcode path — preserving the executor's current
    resilience and keeping gpt2/distilgpt2 byte-identical to today.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        return _derive_model_info(config, model_id)
    except Exception:  # noqa: BLE001 — any failure -> resilient gpt2 fallback
        return _gpt2_hardcode_model_info(model_id)


def catalog_entry_for_config(config: Any, model_id: str) -> Dict[str, Any]:
    """Sprint 1205 — derive a parallax-catalog ModelInfo entry (a dict of kwargs) from a
    HF config. Reuses _derive_model_info (the same logic the local path uses for layer
    slicing), so a production instruct model's num_layers/kv_heads/head_size/etc. are
    correct for the MULTI-HOST scheduler — no hand-authoring. Round-trips through
    ModelInfo(**entry) and satisfies the v1 catalog's required fields."""
    from dataclasses import asdict
    return asdict(_derive_model_info(config, model_id))


def build_parallax_catalog_v1(model_id_to_config: Dict[str, Any]) -> Dict[str, Any]:
    """Sprint 1205 — assemble a v1 parallax model-catalog
    (``{schema_version: "v1", models: {model_id: ModelInfo-kwargs}}``) from
    {model_id: HF-config}. This is the exact shape build_parallax_executor_or_none reads
    from PRSM_PARALLAX_MODEL_CATALOG_FILE."""
    return {
        "schema_version": "v1",
        "models": {
            mid: catalog_entry_for_config(cfg, mid)
            for mid, cfg in model_id_to_config.items()
        },
    }


class _SelfAnchor:
    def __init__(self, node_id: str) -> None:
        self._node_id = node_id

    def lookup(self, node_id: str) -> Optional[str]:
        return ("pk-" + node_id) if node_id == self._node_id else None


class _SelfStakeLookup:
    def __init__(self, node_id: str) -> None:
        self._node_id = node_id

    def get_stake(self, node_id: str) -> int:
        return 10**18 if node_id == self._node_id else 0


class _NoopSubmitter:
    def __call__(self, record: Any) -> None:  # noqa: ARG002
        return None


def build_local_inference_executor(
    node_identity: Any,
    *,
    model_id: str = DEFAULT_LOCAL_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    offline: Optional[bool] = None,
) -> ParallaxScheduledExecutor:
    """Assemble a runnable single-node REAL-inference ``ParallaxScheduledExecutor``.

    The lone GPU is this node — staked, hardware-tier attested (so all privacy
    tiers pass the tier gate), and anchor-registered so the trust stack admits it
    and the allocator places all model layers on it (single-stage). The chain
    executor runs the real HF model in-process. Receipts are signed by
    ``node_identity`` (software-tier crypto — honest scope).
    """
    if node_identity is None or not hasattr(node_identity, "node_id"):
        raise RuntimeError(
            "build_local_inference_executor requires a NodeIdentity (.node_id)"
        )
    node_id = node_identity.node_id
    # Derive ModelInfo ONCE from the real config (sp1133) so the layer count that
    # sizes the node's layer_capacity matches the catalog entry that drives slicing —
    # no second gpt2-default lurking here.
    model_info = _model_info_for(model_id)
    num_layers = model_info.num_layers

    # sp1098 (Domain-03 review F3) — the local executor runs in pure software
    # (TEEType.SOFTWARE), so it must NOT advertise a hardware tier. The prior hard-coded
    # "tier-sgx" silently admitted privacy_tier up to MAXIMUM / Tier B/C confidential
    # work and served it in software with no real confidentiality + no warning. Advertise
    # the honest software tier (tier-none): the TierGateAdapter then refuses confidential
    # tiers on this node. A single-node operator who explicitly wants to exercise the
    # confidential code path in software uses PRSM_PARALLAX_TIER_GATE=advisory, which is
    # the documented escape hatch that WARNS (sp702/sp1084) rather than silently downgrades.
    from prsm.compute.parallax_scheduling.prsm_types import TIER_ATTESTATION_NONE
    gpu = ParallaxGPU(
        node_id=node_id,
        region="local-region",
        layer_capacity=max(num_layers, 16),  # holds the whole model on one node
        stake_amount=10**18,
        tier_attestation=TIER_ATTESTATION_NONE,  # software runtime → no hardware tier
        tflops_fp16=100.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=2000.0,
    )
    pool = [gpu]

    trust = TrustStack(
        anchor_verify=AnchorVerifyAdapter(anchor=_SelfAnchor(node_id)),
        tier_gate=TierGateAdapter(),
        profile_source=StakeWeightedTrustAdapter(
            inner=InMemoryProfileSource(
                snapshots={
                    node_id: ProfileSnapshot(
                        node_id=node_id,
                        layer_latency_ms=10.0,
                        rtt_to_peers={},
                        timestamp_unix=1000.0,
                    ),
                },
            ),
            stake_lookup=_SelfStakeLookup(node_id),
        ),
        consensus_hook=ConsensusMismatchHook(submitter=_NoopSubmitter(), sample_rate=0.0),
    )

    catalog: Dict[str, ModelInfo] = {model_id: model_info}

    def _provider() -> List[ParallaxGPU]:
        return list(pool)

    return ParallaxScheduledExecutor(
        gpu_pool_provider=_provider,
        trust_stack=trust,
        model_catalog=catalog,
        chain_executor=LocalHuggingFaceChainExecutor(
            model_id=model_id, max_tokens=max_tokens, offline=offline),
        node_identity=node_identity,
    )


def _find_local_chain_executor(
    executor: Any,
) -> Optional["LocalHuggingFaceChainExecutor"]:
    """Reach the inner LocalHuggingFaceChainExecutor from a ParallaxScheduledExecutor
    (``._chain_executor``) or accept a chain executor passed directly. Returns None
    when neither is a warmable local executor."""
    if isinstance(executor, LocalHuggingFaceChainExecutor):
        return executor
    inner = getattr(executor, "_chain_executor", None)
    if isinstance(inner, LocalHuggingFaceChainExecutor):
        return inner
    return None


def start_local_prewarm(
    executor: Any, *, enabled: Optional[bool] = None,
) -> Optional[threading.Thread]:
    """sp1302 — kick off a background daemon thread that ``warm()``s the local model
    so the first /compute/inference is fast instead of a 10–30s cold-load hang.

    Returns the started Thread, or None when pre-warm is disabled
    (``PRSM_LOCAL_INFERENCE_PREWARM=0``), the executor is already loaded, or it is
    not a local executor. Model load is BLOCKING (torch), so a thread — not an
    asyncio task — keeps it off both the event loop and the startup path. Fail-soft:
    ``warm()`` swallows its own errors, so the thread can never crash the node."""
    inner = _find_local_chain_executor(executor)
    if inner is None or not _prewarm_enabled(enabled) or inner.is_loaded:
        return None
    t = threading.Thread(
        target=inner.warm, name="local-inference-prewarm", daemon=True)
    t.start()
    return t


def local_inference_readiness(executor: Any) -> Dict[str, Any]:
    """sp1302 — readiness snapshot for status/health surfaces. Returns
    ``{enabled: True, kind: "local", ...describe()}`` for a local executor, else
    ``{enabled: False, kind: None}``. Never triggers a model load."""
    inner = _find_local_chain_executor(executor)
    if inner is None:
        return {"enabled": False, "kind": None}
    return {"enabled": True, "kind": "local", **inner.describe()}
