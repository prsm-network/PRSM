"""Phase 7.1x: ConsensusChallengeSubmitter.

Closes the §8.6 seam from the Phase 7.1 Task 8 review: the
MarketplaceOrchestrator queues minority receipts from consensus
dispatches in `consensus_minority_queue`, but never submits on-chain
`CONSENSUS_MISMATCH` challenges itself. This module is the submitter
service that drains the queue and actually slashes minority providers
via `BatchSettlementRegistry.challengeReceipt`.

Responsibilities:
  - Accept a `ChallengeAttempt` (minority + majority pair from a
    ConsensusShardReceipt, plus both batch_ids and a Merkle proof
    for the majority leaf in its committed batch).
  - Build the `auxData` bytes per the Phase 7.1 `_handleConsensusMismatch`
    layout: `abi.encode(conflictingBatchId, majorityProof, majorityLeaf)`.
  - Sign + broadcast `challengeReceipt(minorityBatchId, minorityLeaf,
    [minority-batch Merkle proof], ReasonCode.CONSENSUS_MISMATCH, aux)`.
  - Surface the outcome as a `ChallengeResult` — success with tx hash,
    or the specific contract revert that fired.
  - Force a gas budget comfortably above Phase 7 §8.7's
    `MIN_SLASH_GAS` so the estimator-binary-search still catches the
    floor but the slash completes inside the forced budget.

Explicit non-responsibilities (deferred to later work):
  - **Batch commit coordination.** The caller is responsible for
    ensuring both the minority and majority batches are committed
    on-chain before calling `submit_one`. The submitter will surface
    `ConflictingBatchNotCommitted` cleanly if a batch isn't ready.
  - **Persistence.** The drained queue entries live in-memory in the
    caller. If this process crashes mid-submission, those receipts
    are lost. Adding a SQLite backing store is a follow-up.
  - **Retry / backoff.** The submitter returns per-attempt results;
    the caller decides whether to retry. No built-in queuing.
  - **Proof construction.** The caller constructs the Merkle proof for
    the majority leaf. For 1-leaf batches the proof is empty; for
    multi-leaf batches the caller owns the tree. The submitter just
    relays bytes to the contract.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from base64 import b64decode

try:
    from web3 import Web3
    from eth_abi import encode as abi_encode
    from eth_account import Account
    from eth_utils import keccak
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)
from prsm.economy.web3.stake_manager import ReasonCode

logger = logging.getLogger(__name__)


# Minimum gas to send with a CONSENSUS_MISMATCH challenge tx. The
# registry now requires gasleft() >= MIN_SLASH_GAS (150_000) before the
# slash try/catch. 1_000_000 leaves plenty of headroom for the full
# path (Merkle verify + binding checks + slash + event emission).
DEFAULT_CHALLENGE_GAS = 1_000_000


@dataclass(frozen=True)
class ReceiptLeafFields:
    """On-chain-encoded ReceiptLeaf tuple fields.

    Matches `BatchSettlementRegistry.ReceiptLeaf` (sprint 347 —
    L2 audit C-INT-01 added the trailing `signingMessageHash`):
      (bytes32, uint32, bytes32, bytes32, bytes32, uint64,
       uint128, bytes32, bytes32)

    The caller is responsible for deriving these from the Python
    `ShardExecutionReceipt` — see `from_python_receipt` below for the
    canonical conversion.
    """
    job_id_hash: bytes
    shard_index: int
    provider_id_hash: bytes
    provider_pubkey_hash: bytes
    output_hash: bytes
    executed_at_unix: int
    value_ftns_wei: int
    signature_hash: bytes
    # Sprint 347 — L2 audit C-INT-01 fix. Binds the INVALID_
    # SIGNATURE challenge path to a leaf-committed message hash
    # so the challenger can't re-target verification to an
    # attacker-chosen message. The submitter doesn't use the
    # CONSENSUS_MISMATCH path, but the field is part of the
    # struct selector and must be present for the on-chain
    # ReceiptLeaf tuple to encode/decode cleanly.
    signing_message_hash: bytes

    @classmethod
    def from_python_receipt(
        cls,
        receipt: ShardExecutionReceipt,
        value_ftns_wei: int,
    ) -> "ReceiptLeafFields":
        """Convert a Python `ShardExecutionReceipt` into the on-chain
        ReceiptLeaf fields.

        Hashing rules MUST match the CANONICAL committed-leaf convention in
        ``prsm.settlement.merkle.batched_receipt_to_leaf`` (which is what the
        production batch-commit path folds the on-chain merkleRoot over), so a
        challenge leaf re-hashes to a value that is actually in the committed
        tree:

          - job_id_hash = keccak256(job_id.utf8)
          - provider_id_hash = keccak256(provider_id.utf8)
          - provider_pubkey_hash = keccak256(b64decode(provider_pubkey_b64))
          - output_hash = bytes.fromhex(receipt.output_hash)  (hex str → 32 bytes)
          - signature_hash = keccak256(b64decode(signature))
          - signing_message_hash = keccak256(signing_message.utf8)
            where signing_message is the canonical signing-payload
            (build_receipt_signing_payload). The receipt itself
            doesn't carry the message — we reconstruct it from
            (job_id, shard_index, output_hash, executed_at_unix).

        sp1165 FIX: provider_pubkey_hash + signature_hash previously hashed the
        base64 STRING bytes (keccak(utf8(b64))) — diverging from the committed
        convention (keccak(b64decode(...))), so a CONSENSUS_MISMATCH challenge
        hard-reverted ``InvalidMerkleProof`` against any real committed batch
        (the contract verifies the proof on the re-hashed leaf BEFORE the reason
        dispatch). Now aligned with merkle.batched_receipt_to_leaf.

        `value_ftns_wei` comes from the orchestrator's per-provider
        escrow amount; the receipt itself doesn't carry a price.
        """
        if not HAS_WEB3:
            raise RuntimeError("web3 required for leaf-field hashing")
        output_bytes = bytes.fromhex(receipt.output_hash)
        if len(output_bytes) != 32:
            raise ValueError(
                f"receipt.output_hash must be 32 bytes hex (got "
                f"{len(output_bytes)})"
            )
        signing_payload = (
            f"{receipt.job_id}||{receipt.shard_index}||"
            f"{receipt.output_hash}||{receipt.executed_at_unix}"
        ).encode("utf-8")
        return cls(
            job_id_hash=bytes(keccak(receipt.job_id.encode("utf-8"))),
            shard_index=receipt.shard_index,
            provider_id_hash=bytes(keccak(receipt.provider_id.encode("utf-8"))),
            provider_pubkey_hash=bytes(
                keccak(b64decode(receipt.provider_pubkey_b64))
            ),
            output_hash=output_bytes,
            executed_at_unix=receipt.executed_at_unix,
            value_ftns_wei=value_ftns_wei,
            signature_hash=bytes(keccak(b64decode(receipt.signature))),
            signing_message_hash=bytes(keccak(signing_payload)),
        )

    def to_tuple(self) -> Tuple:
        """Return the fields as the tuple web3.py expects for the
        contract's `ReceiptLeaf calldata` argument."""
        return (
            self.job_id_hash, self.shard_index, self.provider_id_hash,
            self.provider_pubkey_hash, self.output_hash,
            self.executed_at_unix, self.value_ftns_wei,
            self.signature_hash, self.signing_message_hash,
        )


@dataclass(frozen=True)
class ChallengeAttempt:
    """One CONSENSUS_MISMATCH challenge worth of inputs.

    Built by the caller after batch commit: the caller knows both
    batch_ids, the full `ReceiptLeafFields` for each, and any Merkle
    proofs needed for multi-leaf batches (empty for 1-leaf).
    """
    minority_batch_id: bytes
    minority_leaf: ReceiptLeafFields
    minority_proof: List[bytes]   # proof within the minority's batch
    majority_batch_id: bytes
    majority_leaf: ReceiptLeafFields
    majority_proof: List[bytes]   # proof within the majority's batch


@dataclass(frozen=True)
class ChallengeResult:
    """Outcome of one challenge attempt."""
    success: bool
    tx_hash_hex: Optional[str]
    error_type: Optional[str]   # exception class name, None on success
    error_message: Optional[str]


@dataclass(frozen=True)
class ConsensusDryRunResult:
    """Read-only pre-flight verdict — would the CONSENSUS_MISMATCH challenge succeed if
    broadcast? Mirrors invalid_signature_submitter.DryRunResult (duck-typed: would_succeed +
    revert_reason) without a marketplace->settlement import."""
    would_succeed: bool
    revert_reason: Optional[str] = None


def encode_consensus_mismatch_aux(
    *, majority_batch_id: bytes, majority_proof: List[bytes], majority_leaf_tuple: Tuple,
) -> bytes:
    """The CONSENSUS_MISMATCH ``auxData`` the on-chain ``_handleConsensusMismatch`` decodes:
       abi.encode(bytes32 conflictingBatchId, bytes32[] majorityProof, ReceiptLeaf majorityLeaf).
    Module-level so the sp1165 assembler reuses the EXACT layout the submitter broadcasts (no
    drift). ``majority_leaf_tuple`` is ``ReceiptLeafFields.to_tuple()``."""
    if not HAS_WEB3:
        raise RuntimeError("web3 required for consensus auxData encoding")
    return bytes(abi_encode(
        [
            "bytes32",
            "bytes32[]",
            "(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32,bytes32)",
        ],
        [bytes(majority_batch_id), [bytes(p) for p in majority_proof],
         tuple(majority_leaf_tuple)],
    ))


class ConsensusChallengeSubmitter:
    """Sync Web3 client for CONSENSUS_MISMATCH challenges.

    One instance per challenger keypair. Shares the nonce-lock pattern
    established by `ProvenanceRegistryClient` and `StakeManagerClient`
    (see `OPERATOR_GUIDE.md` §On-chain Keypairs for the one-keypair-
    per-process invariant).
    """

    CHALLENGE_RECEIPT_ABI = [{
        "inputs": [
            {"name": "batchId", "type": "bytes32"},
            {
                # Sprint 347 — ReceiptLeaf gained
                # `signingMessageHash` (L2 audit C-INT-01).
                "components": [
                    {"name": "jobIdHash", "type": "bytes32"},
                    {"name": "shardIndex", "type": "uint32"},
                    {"name": "providerIdHash", "type": "bytes32"},
                    {"name": "providerPubkeyHash", "type": "bytes32"},
                    {"name": "outputHash", "type": "bytes32"},
                    {"name": "executedAtUnix", "type": "uint64"},
                    {"name": "valueFtns", "type": "uint128"},
                    {"name": "signatureHash", "type": "bytes32"},
                    {"name": "signingMessageHash", "type": "bytes32"},
                ],
                "name": "leaf",
                "type": "tuple",
            },
            {"name": "merkleProof", "type": "bytes32[]"},
            {"name": "reason", "type": "uint8"},
            {"name": "auxData", "type": "bytes"},
        ],
        "name": "challengeReceipt",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }]

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        registry_address: Optional[str] = None,
        private_key: Optional[str] = None,
        gas_budget: int = DEFAULT_CHALLENGE_GAS,
        *,
        web3: object = None,
        registry: object = None,
        account: object = None,
    ) -> None:
        self._tx_lock = threading.Lock()
        self.gas_budget = gas_budget
        # Injection seam (sp1165) — mirrors invalid_signature_submitter.ChallengeSubmitter:
        # inject (web3, registry, account) for tests OR a KEYLESS read-only dry-run (the
        # ``account`` may be a from-address-only stub with no signing key, since dry_run does
        # a static eth_call). The live broadcast path still needs the rpc/key build below.
        if registry is not None and account is not None:
            self.web3 = web3
            self.registry = registry
            self._account = account
            self.registry_address = getattr(registry, "address", None)
            return
        if not HAS_WEB3:
            raise RuntimeError(
                "web3 package is required (pip install web3 eth-account)"
            )
        if not (rpc_url and registry_address and private_key):
            raise ValueError(
                "rpc_url, registry_address and private_key are required for the live path "
                "(or inject registry+account for a keyless dry-run / tests)"
            )
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.registry_address = Web3.to_checksum_address(registry_address)
        self.registry = self.web3.eth.contract(
            address=self.registry_address,
            abi=self.CHALLENGE_RECEIPT_ABI,
        )
        self._account = Account.from_key(private_key)

    @property
    def address(self) -> str:
        return self._account.address

    # ── read-only pre-flight (NOT a broadcast) ─────────────────

    def dry_run(self, challenge: object) -> ConsensusDryRunResult:
        """Static eth_call of ``challengeReceipt(*challenge.to_call_args())`` against current
        chain state — no tx, no gas, no state change. ``challenge`` is a
        ``ConsensusMismatchChallenge`` (reason CONSENSUS_MISMATCH; ``to_call_args()`` already
        carries the consensus auxData). Returns whether the challenge WOULD succeed. This is
        the verify-before-broadcast pre-flight the assistant/operator runs; the actual
        broadcast (``submit_one``) stays the separate, user-gated step."""
        try:
            self.registry.functions.challengeReceipt(
                *challenge.to_call_args()
            ).call({"from": self.address})
            return ConsensusDryRunResult(would_succeed=True)
        except Exception as exc:  # noqa: BLE001 — surface the revert reason, don't raise
            return ConsensusDryRunResult(would_succeed=False, revert_reason=str(exc))

    # ── Writes ─────────────────────────────────────────────────

    def submit_one(self, attempt: ChallengeAttempt) -> ChallengeResult:
        """Submit a single CONSENSUS_MISMATCH challenge on-chain.

        Returns a `ChallengeResult` capturing success + tx hash, or the
        contract revert that fired. Never raises — the caller receives
        a uniform result shape and decides whether to retry, escalate,
        or drop.
        """
        try:
            aux = self._encode_aux(attempt)
            with self._tx_lock:
                tx = self.registry.functions.challengeReceipt(
                    attempt.minority_batch_id,
                    attempt.minority_leaf.to_tuple(),
                    attempt.minority_proof,
                    int(ReasonCode.CONSENSUS_MISMATCH),
                    aux,
                ).build_transaction(self._tx_overrides())
                tx_hash_hex, _ = self._sign_and_send(tx)
            return ChallengeResult(
                success=True, tx_hash_hex=tx_hash_hex,
                error_type=None, error_message=None,
            )
        except (BroadcastFailedError, OnChainPendingError,
                OnChainRevertedError) as exc:
            logger.warning(
                f"consensus challenge failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return ChallengeResult(
                success=False,
                tx_hash_hex=getattr(exc, "tx_hash", None),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        except Exception as exc:
            # Catch-all so the caller's loop through a queue never
            # aborts on one broken attempt. Log loudly so the bug
            # surfaces.
            logger.error(
                f"consensus challenge raised unexpected {type(exc).__name__}: "
                f"{exc}"
            )
            return ChallengeResult(
                success=False, tx_hash_hex=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def submit_batch(
        self, attempts: List[ChallengeAttempt],
    ) -> List[ChallengeResult]:
        """Submit a list of attempts sequentially. Returns a parallel
        list of results. One failure does NOT abort the loop."""
        return [self.submit_one(a) for a in attempts]

    # ── Internals ──────────────────────────────────────────────

    def _encode_aux(self, attempt: ChallengeAttempt) -> bytes:
        """Mirror `_handleConsensusMismatch`'s auxData layout. Delegates to the shared
        module-level ``encode_consensus_mismatch_aux`` so the sp1165 prepare assembler and
        this submitter encode auxData IDENTICALLY (single source of truth)."""
        return encode_consensus_mismatch_aux(
            majority_batch_id=attempt.majority_batch_id,
            majority_proof=attempt.majority_proof,
            majority_leaf_tuple=attempt.majority_leaf.to_tuple(),
        )

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending"
            ),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
            # Force enough gas to clear MIN_SLASH_GAS + full slash path.
            # Without this, estimateGas can under-budget per Phase 7 §8.7
            # — though the new floor makes under-funding revert cleanly
            # instead of silently swallowing the slash.
            "gas": self.gas_budget,
        }

    def _sign_and_send(self, tx: dict) -> Tuple[str, "TransferStatus"]:
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )
        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise BroadcastFailedError(f"broadcast failed: {exc}") from exc

        tx_hash_hex = "0x" + tx_hash_bytes.hex()
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash_bytes, timeout=120,
            )
        except Exception as exc:
            raise OnChainPendingError(
                f"broadcast OK but receipt unknown: {exc}",
                tx_hash=tx_hash_hex,
            ) from exc

        if receipt.status != 1:
            raise OnChainRevertedError(f"challenge tx reverted: {tx_hash_hex}")

        return tx_hash_hex, TransferStatus.CONFIRMED


# ── Runner: drives queue → submit → mark_* ─────────────────────


# Phase 7.1x.next — Pre-audit hardening (review gate P1):
# Explicit allowlist of retryable errors. Default-deny is safer than
# default-retry because unexpected bugs (KeyError in encoder, runtime
# type errors, etc.) surface in submit_one's catch-all as failure
# results with the bug's exception class as error_type. Without an
# allowlist those rows would bounce between SUBMITTABLE and mark_failed
# forever, accumulating attempts but never advancing.
#
# Only BroadcastFailedError is genuinely transient — the tx was never
# signed/sent, retrying is safe and correct. Every other failure path
# is terminal:
#   - OnChainRevertedError: contract said no. Retrying reproduces the
#     revert.
#   - OnChainPendingError: broadcast succeeded but receipt unknown.
#     UNSAFE to auto-retry (could double-submit). Operator reconciles
#     via the recorded tx_hash.
#   - Any other type: unexpected bug. Stop bleeding ops budget on it
#     and surface as terminal so a human investigates.
_RETRYABLE_ERROR_TYPES = frozenset({"BroadcastFailedError"})

# Absolute attempt cap even for retryable errors. Without this, a
# misconfigured RPC endpoint could cause a submittable row to retry
# indefinitely. After N attempts we terminal-fail the row so operator
# can investigate. Tuned conservatively — 5 is plenty for transient
# blips; chronic failures should surface.
DEFAULT_MAX_ATTEMPTS = 5


def process_submittable_queue(
    queue,
    submitter: ConsensusChallengeSubmitter,
    limit: int = 100,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    claimant_id: Optional[str] = None,
    lease_seconds: Optional[float] = None,
) -> List[ChallengeResult]:
    """Phase 7.1x.next runner loop: pull all SUBMITTABLE rows from the
    queue, fire CONSENSUS_MISMATCH challenges via the submitter, update
    row statuses based on outcome. Returns the list of results in the
    same order the queue returned rows.

    Safe to call repeatedly from a scheduler (cron / asyncio.sleep
    loop). Idempotent once a row transitions to a terminal state.

    Accepts the queue as a parameter rather than importing
    `ConsensusChallengeQueue` directly so this module doesn't force a
    sqlite3 dependency on callers who only want the submitter API.

    max_attempts caps retries per row even for retryable errors —
    prevents a misconfigured RPC endpoint from burning ops budget on
    one row indefinitely.

    Phase 7.1x.next+ §5.3: pass claimant_id to opt into multi-runner-
    safe mode. The runner atomically claims SUBMITTABLE rows with a
    time-bounded lease, preventing two concurrent runners against the
    same DB from double-submitting. When claimant_id is None (the
    single-runner default), the legacy list_submittable path is used —
    faster, no claim overhead, no safety guarantee across processes.
    """
    if claimant_id is not None:
        claim_kwargs = {"claimant_id": claimant_id, "limit": limit}
        if lease_seconds is not None:
            claim_kwargs["lease_seconds"] = lease_seconds
        rows = queue.claim_submittable(**claim_kwargs)
    else:
        rows = queue.list_submittable(limit=limit)
    results: List[ChallengeResult] = []
    for row in rows:
        attempt = ChallengeAttempt(
            minority_batch_id=row.minority_batch_id,
            minority_leaf=ReceiptLeafFields.from_python_receipt(
                row.minority_receipt,
                value_ftns_wei=row.value_ftns_per_provider_wei,
            ),
            minority_proof=[],   # 1-leaf-per-provider batches in MVP
            majority_batch_id=row.majority_batch_id,
            majority_leaf=ReceiptLeafFields.from_python_receipt(
                row.majority_receipt,
                value_ftns_wei=row.value_ftns_per_provider_wei,
            ),
            majority_proof=[],
        )
        result = submitter.submit_one(attempt)
        if result.success:
            queue.mark_submitted(row.row_id, result.tx_hash_hex)
        else:
            # Terminal unless the error is explicitly retryable AND the
            # row hasn't exhausted its attempt budget. submit_one
            # increments attempts via mark_failed, so a row at attempts
            # == max_attempts-1 will hit max_attempts after this call
            # and should terminal now.
            is_retryable_type = result.error_type in _RETRYABLE_ERROR_TYPES
            under_cap = row.attempts + 1 < max_attempts
            terminal = not (is_retryable_type and under_cap)
            queue.mark_failed(
                row.row_id,
                result.error_type or "UnknownError",
                result.error_message or "",
                terminal=terminal,
            )
        results.append(result)
    return results


__all__ = [
    "ConsensusChallengeSubmitter",
    "ChallengeAttempt",
    "ChallengeResult",
    "ReceiptLeafFields",
    "DEFAULT_CHALLENGE_GAS",
    "process_submittable_queue",
]
