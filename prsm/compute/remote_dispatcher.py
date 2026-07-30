"""Phase 2: RemoteShardDispatcher.

Plugs into TensorParallelExecutor's `remote_dispatcher` slot. Owns
the full round-trip for a single shard: size check → peer resolve →
escrow create → MSG_DIRECT send → await response → verify receipt →
escrow release → return output. Refunds escrow on any failure.
Retries on timeout up to max_retries; falls back to local execution
if local_fallback is wired.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from prsm.compute.model_sharding.models import ModelShard, PipelineStakeTier
from prsm.compute.shard_receipt import VerificationStrategy
from prsm.node.transport import MSG_DIRECT, P2PMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchResult:
    """Augmented result from `dispatch_with_receipt`. Returned when the
    caller needs the verified Phase 2 receipt alongside the numerical
    output — e.g., for Phase 3.1 batched-settlement accumulation.

    The plain `dispatch()` method discards the receipt and returns just
    the output for backwards-compatibility with Phase 2 callers.
    """
    output: np.ndarray
    receipt: Dict[str, Any]            # Phase 2 ShardExecutionReceipt dict form
    provider_node_id: str              # node_id that served this shard
    escrow_amount_ftns: float          # escrow amount that will be released


class ShardTooLargeError(ValueError):
    """Shard tensor exceeds max_shard_bytes."""


class PeerNotConnectedError(RuntimeError):
    """Target node_id is not a connected peer."""


class ShardDispatchError(RuntimeError):
    """Generic dispatch failure after escrow refund."""


class EscrowCreationFailedError(RuntimeError):
    """PaymentEscrow.create_escrow returned None (e.g., insufficient
    requester balance). Dispatch must NOT execute the shard without a
    funded escrow — that would break the Phase 2 pay-for-work guarantee."""


class EscrowReleaseFailedError(RuntimeError):
    """PaymentEscrow.release_escrow returned None after a successful
    shard execution. The escrow is still PENDING on the ledger; funds
    have NOT moved. Callers must reconcile manually — dispatch does NOT
    auto-refund here because the compute itself succeeded and the right
    recovery is a retry of release (idempotent) or an operator-initiated
    reconciliation, not a refund to the requester."""


class MissingAttestationError(RuntimeError):
    """A dispatch requested TEE attestation (require_tee_attestation=True)
    but the returned receipt did not include a tee_attestation field.

    Per Phase 3 design §7.2: a provider that claimed tee_capable in its
    marketplace listing but fails to include a quote in its receipt is
    effectively lying about its TEE support. The dispatcher refunds
    escrow and surfaces this distinctly from ShardDispatchError so the
    marketplace orchestrator can drop the provider's reputation below
    the simple "failed dispatch" signal."""


class ShardPreemptedError(RuntimeError):
    """Provider reported status=preempted (e.g., cloud spot-instance
    eviction). Phase 2.1 Line Item A contract:

      - Honest-work failure, NOT malicious. Callers MUST NOT tag the
        provider as misbehaving or feed this into any future slashing
        signal (Phase 7).
      - Escrow is refunded (no compute output was delivered for this
        shard) so the requester can re-dispatch elsewhere.
      - Re-routing is the caller's responsibility — dispatcher does
        not auto-retry on a different node because node selection is
        a Line Item B concern (topology randomization).
    """

    def __init__(self, shard_index: int, node_id: str, reason: str = ""):
        self.shard_index = shard_index
        self.node_id = node_id
        self.reason = reason
        super().__init__(
            f"shard {shard_index} preempted on node {node_id[:12]}…: {reason}"
        )


def _escrow_job_id(job_id: str, shard_index: int) -> str:
    """Compose the unique escrow-tracking key for one shard-exec lease.

    PaymentEscrow is keyed by job_id; one model-exec job fans out to
    N shards, each needing its own escrow, so we namespace by shard.

    sp1488 — the uniqueness this key promises is only as good as ``job_id``.
    An empty one collapses every concurrent job's shard 0 to ``":shard:0"``,
    so callers must validate before composing (see
    ``validate_dispatch_assignment``).
    """
    if not job_id:
        raise InvalidDispatchAssignment(
            "refusing to build an escrow key from an EMPTY job_id: every "
            "concurrent job's shard would collapse onto ':shard:<n>' and share "
            "one escrow, so one job's release would settle another job's funds"
        )
    return f"{job_id}:shard:{shard_index}"


class InvalidDispatchAssignment(ValueError):
    """The assignment cannot be paid for safely. Nothing was dispatched.

    Raised INSTEAD of substituting a default, because every plausible default
    here is a money bug rather than a degraded mode.
    """


def validate_dispatch_assignment(assignment: dict) -> tuple:
    """sp1488 — fail closed on an assignment that cannot be paid for correctly.

    The adapter that bridges TensorParallelExecutor to RemoteShardDispatcher used
    ``assignment.get(k, <default>)`` for the three fields that decide WHO gets
    paid, HOW MUCH, and WHICH ESCROW — and each default is a distinct money bug,
    not a graceful fallback:

      * ``job_id`` -> ``""``   collapses the escrow key to ``":shard:<n>"``, so
        two concurrent jobs share one escrow and one job's release settles the
        other's funds.
      * ``escrow_amount_ftns`` -> ``1.0`` invents a price nobody quoted or
        agreed to, silently over- or under-paying against the real quote.
      * ``node_id`` -> ``""``  dispatches to no one while still creating (and
        potentially stranding) an escrow.

    A missing field means the caller is broken, and the only safe response is to
    dispatch nothing: an unpaid provider or a mis-settled escrow cannot be undone
    from here, while refusing costs one failed job.

    Returns ``(node_id, job_id, escrow_amount_ftns)``.
    """
    node_id = (assignment.get("node_id") or "").strip()
    job_id = (assignment.get("job_id") or "").strip()
    if not node_id:
        raise InvalidDispatchAssignment(
            "assignment has no node_id — refusing to dispatch: there is no "
            "provider to pay, and creating the escrow anyway would strand funds"
        )
    if not job_id:
        raise InvalidDispatchAssignment(
            "assignment has no job_id — refusing to dispatch: the escrow key "
            f"would be ':shard:<n>', which every CONCURRENT job also produces, "
            "so one job's release would settle another job's escrow"
        )

    if "escrow_amount_ftns" not in assignment:
        raise InvalidDispatchAssignment(
            "assignment has no escrow_amount_ftns — refusing to dispatch rather "
            "than inventing a price. The amount must come from the accepted "
            "quote; a default silently pays something nobody agreed to"
        )
    try:
        amount = float(assignment["escrow_amount_ftns"])
    except (TypeError, ValueError) as exc:
        raise InvalidDispatchAssignment(
            f"escrow_amount_ftns is not a number: "
            f"{assignment['escrow_amount_ftns']!r}"
        ) from exc
    if amount != amount or amount in (float("inf"), float("-inf")):
        # NaN/Inf survive float() and would poison every downstream balance —
        # the sp1468 DAGLedger lesson.
        raise InvalidDispatchAssignment(
            f"escrow_amount_ftns is not finite: {amount!r}")
    if amount <= 0:
        raise InvalidDispatchAssignment(
            f"escrow_amount_ftns must be positive, got {amount} — a zero or "
            "negative escrow means the provider works for nothing"
        )
    return node_id, job_id, amount


class RemoteShardDispatcher:
    """Dispatches a shard to a remote provider via MSG_DIRECT.

    The executor's remote_dispatcher slot expects a callable
    `(shard, input_data, assignment) -> Awaitable[dict]`. Task 6 wires
    this class into that slot by adapting dispatch()'s return shape.
    """

    def __init__(
        self,
        identity,
        transport,
        payment_escrow,
        verification_strategy: VerificationStrategy,
        default_timeout: float = 30.0,
        max_retries: int = 1,
        max_shard_bytes: int = 10 * 1024 * 1024,
        local_fallback: Optional[Callable] = None,
    ):
        self.identity = identity
        self.transport = transport
        self.payment_escrow = payment_escrow
        self.verification_strategy = verification_strategy
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.max_shard_bytes = max_shard_bytes
        self.local_fallback = local_fallback

        self._pending: Dict[str, asyncio.Future] = {}

        self.transport.on_message(MSG_DIRECT, self._on_direct_message)

    async def dispatch(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        node_id: str,
        job_id: str,
        stake_tier: PipelineStakeTier,
        escrow_amount_ftns: float,
        require_tee_attestation: bool = False,
    ) -> np.ndarray:
        """Dispatch shard to node_id and return the output tensor.
        Raises on unrecoverable failure after escrow refund.

        This is the Phase 2 API — returns just the output for
        backwards-compatible callers. Phase 3.1 callers needing the
        verified receipt for batched-settlement accumulation should
        use `dispatch_with_receipt` instead.
        """
        result = await self.dispatch_with_receipt(
            shard=shard,
            input_tensor=input_tensor,
            node_id=node_id,
            job_id=job_id,
            stake_tier=stake_tier,
            escrow_amount_ftns=escrow_amount_ftns,
            require_tee_attestation=require_tee_attestation,
        )
        return result.output

    async def dispatch_with_receipt(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        node_id: str,
        job_id: str,
        stake_tier: PipelineStakeTier,
        escrow_amount_ftns: float,
        require_tee_attestation: bool = False,
    ) -> DispatchResult:
        """Phase 3.1 variant: returns DispatchResult with both the
        numerical output AND the verified receipt dict, suitable for
        forwarding to a BatchSettlementClient for on-chain accumulation.

        On fallback paths (size-too-large or peer-not-connected with
        a local_fallback wired), the returned DispatchResult has an
        empty receipt dict — fallback execution produces no on-chain-
        settleable evidence, so there's nothing to accumulate. Callers
        MUST check `bool(result.receipt)` before forwarding to the
        settlement layer.
        """
        if len(shard.tensor_data) > self.max_shard_bytes:
            if self.local_fallback is not None:
                logger.warning(
                    f"shard {shard.shard_index} size "
                    f"{len(shard.tensor_data)} exceeds max "
                    f"{self.max_shard_bytes}; falling back to local"
                )
                output = await self._call_fallback(shard, input_tensor)
                return DispatchResult(
                    output=output, receipt={},
                    provider_node_id=node_id,
                    escrow_amount_ftns=0.0,
                )
            raise ShardTooLargeError(
                f"shard {shard.shard_index} size "
                f"{len(shard.tensor_data)} exceeds max {self.max_shard_bytes}"
            )

        peer = self.transport.get_peer(node_id)
        if peer is None:
            if self.local_fallback is not None:
                logger.warning(f"peer {node_id} not connected; falling back")
                output = await self._call_fallback(shard, input_tensor)
                return DispatchResult(
                    output=output, receipt={},
                    provider_node_id=node_id,
                    escrow_amount_ftns=0.0,
                )
            raise PeerNotConnectedError(
                f"node_id {node_id!r} is not a connected peer"
            )

        escrow_job_id = _escrow_job_id(job_id, shard.shard_index)
        escrow_entry = await self.payment_escrow.create_escrow(
            job_id=escrow_job_id,
            amount=escrow_amount_ftns,
            requester_id=self.identity.node_id,
        )
        # P1: escrow MUST be funded before we dispatch. If create_escrow
        # returned None (insufficient balance, ledger failure, etc.), the
        # pay-for-work guarantee is broken — refuse to execute.
        if escrow_entry is None:
            raise EscrowCreationFailedError(
                f"failed to create escrow for shard {shard.shard_index} "
                f"of job {job_id!r} (requester={self.identity.node_id[:12]}…, "
                f"amount={escrow_amount_ftns} FTNS)"
            )

        output: Optional[np.ndarray] = None
        receipt: Optional[Dict[str, Any]] = None
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                output, receipt = await self._dispatch_once(
                    shard=shard,
                    input_tensor=input_tensor,
                    node_id=node_id,
                    job_id=job_id,
                    stake_tier=stake_tier,
                    escrow_job_id=escrow_job_id,
                    require_tee_attestation=require_tee_attestation,
                )
                break
            except asyncio.TimeoutError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.info(
                        f"shard {shard.shard_index} dispatch to {node_id} "
                        f"timed out; retry {attempt + 1}/{self.max_retries}"
                    )
                    continue
                break
            except ShardPreemptedError as exc:
                # Honest-work failure — same provider won't help. Exit
                # the retry loop immediately so the outer refund path
                # runs and the caller can re-dispatch elsewhere.
                last_exc = exc
                break
            except MissingAttestationError as exc:
                # Provider claimed TEE support but omitted the quote.
                # No retry on same provider — they're lying.
                last_exc = exc
                break
            except ShardDispatchError as exc:
                last_exc = exc
                break
            except Exception as exc:
                last_exc = exc
                break

        # SUCCESS path: compute completed; attempt escrow release. If
        # release fails, the escrow is still PENDING on the ledger — do
        # NOT auto-refund (compute was honest work), surface the release
        # failure for operator reconciliation.
        if output is not None:
            release_tx = await self.payment_escrow.release_escrow(
                escrow_job_id, node_id,
            )
            if release_tx is None:
                logger.error(
                    f"shard {shard.shard_index} executed successfully "
                    f"but escrow release failed for {escrow_job_id!r}; "
                    f"escrow remains PENDING — reconciliation required. "
                    f"NOT auto-refunding."
                )
                raise EscrowReleaseFailedError(
                    f"shard {shard.shard_index} executed successfully "
                    f"but escrow release failed for {escrow_job_id!r}; "
                    f"reconciliation required"
                )
            return DispatchResult(
                output=output,
                receipt=receipt or {},
                provider_node_id=node_id,
                escrow_amount_ftns=escrow_amount_ftns,
            )

        # FAILURE path: compute did not complete. Refund escrow, then
        # fall back to local (if wired) or re-raise the failure.
        await self.payment_escrow.refund_escrow(
            escrow_job_id, reason=str(last_exc),
        )

        if isinstance(last_exc, asyncio.TimeoutError) and self.local_fallback is not None:
            logger.warning(
                f"shard {shard.shard_index} dispatch failed after "
                f"{self.max_retries} retries; falling back to local"
            )
            fallback_output = await self._call_fallback(shard, input_tensor)
            return DispatchResult(
                output=fallback_output, receipt={},
                provider_node_id=node_id, escrow_amount_ftns=0.0,
            )

        # Preemption is an honest-work signal — re-raise as-is so
        # caller can distinguish from malicious/timeout failures and
        # re-dispatch to a different node without any reputation hit.
        if isinstance(last_exc, ShardPreemptedError):
            raise last_exc

        if isinstance(last_exc, MissingAttestationError):
            raise last_exc

        if isinstance(last_exc, ShardDispatchError):
            raise last_exc

        raise ShardDispatchError(
            f"shard {shard.shard_index} dispatch failed: {last_exc}"
        ) from last_exc

    async def _dispatch_once(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        node_id: str,
        job_id: str,
        stake_tier: PipelineStakeTier,
        escrow_job_id: str,
        require_tee_attestation: bool = False,
    ) -> "tuple[np.ndarray, Dict[str, Any]]":
        """Single dispatch attempt. Raises asyncio.TimeoutError if no
        response arrives within default_timeout."""
        request_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        deadline = int(time.time()) + max(1, int(self.default_timeout) + 1)

        stake_label = (
            stake_tier.label if hasattr(stake_tier, "label")
            else str(stake_tier)
        )

        payload = {
            "subtype": "shard_execute_request",
            "job_id": job_id,
            "shard_index": shard.shard_index,
            "tensor_data_b64": base64.b64encode(shard.tensor_data).decode(),
            "tensor_shape": list(shard.tensor_shape),
            "tensor_dtype": "float64",
            "input_b64": base64.b64encode(input_tensor.tobytes()).decode(),
            "input_shape": list(input_tensor.shape),
            "input_dtype": str(input_tensor.dtype),
            "checksum": shard.checksum,
            "stake_tier": stake_label,
            "escrow_tx_id": escrow_job_id,
            "deadline_unix": deadline,
            "request_id": request_id,
            "requester_pubkey_b64": self.identity.public_key_b64,
        }
        msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload=payload,
        )

        try:
            await self.transport.send_to_peer(node_id, msg)
            response = await asyncio.wait_for(
                future, timeout=self.default_timeout
            )
        finally:
            self._pending.pop(request_id, None)

        status = response.get("status")
        if status == "preempted":
            raise ShardPreemptedError(
                shard_index=shard.shard_index,
                node_id=node_id,
                reason=str(response.get("error") or "cloud-provider eviction"),
            )
        if status != "completed":
            raise ShardDispatchError(
                f"provider returned status={status!r} "
                f"error={response.get('error')!r}"
            )

        output_bytes = base64.b64decode(response["output_b64"])
        output_shape = response["output_shape"]
        output_dtype = np.dtype(response["output_dtype"])
        output = np.frombuffer(output_bytes, dtype=output_dtype).reshape(output_shape)

        receipt = response.get("receipt", {})
        # P2: pass the node_id we dispatched to so the verifier binds
        # the receipt to the intended provider. Otherwise a receipt
        # could claim 'provider A' but carry B's pubkey+sig and still
        # verify (self-authenticating against whatever pubkey it carries).
        verified = await self.verification_strategy.verify(
            receipt, output_bytes, expected_provider_id=node_id,
        )
        if not verified:
            raise ShardDispatchError(
                f"receipt verification failed for shard {shard.shard_index}"
            )

        if require_tee_attestation and not receipt.get("tee_attestation"):
            raise MissingAttestationError(
                f"shard {shard.shard_index} dispatched with "
                f"require_tee_attestation=True but provider "
                f"{node_id[:12]}… returned a receipt with no "
                f"tee_attestation field"
            )

        return output, receipt

    async def _on_direct_message(self, msg, peer) -> None:
        """Route shard_execute_response to pending futures."""
        if msg.payload.get("subtype") != "shard_execute_response":
            return
        request_id = msg.payload.get("request_id")
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(msg.payload)

    async def _call_fallback(
        self, shard: ModelShard, input_tensor: np.ndarray,
    ) -> np.ndarray:
        """Invoke local_fallback. Supports both sync and async callables."""
        result = self.local_fallback(shard, input_tensor)
        if asyncio.iscoroutine(result):
            return await result
        return result
