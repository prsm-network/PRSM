"""Storage proof-of-retrievability challenge/response flow.

Per docs/2026-04-22-phase7-storage-design-plan.md §5.2, §6 Task 4.

Extracts the off-chain challenge/response protocol from the legacy
`prsm/node/storage_proofs.py` into a focused module that (a) operates on
fixed-size-chunk Merkle trees over stored shards, (b) issues deterministic
challenges with replay-protection nonces, (c) verifies responses in pure
Python, and (d) escalates verification failures via an injected on-chain
slash hook (prsm.storage.proof.OnChainSlashHook -> StorageSlashing.sol
Task 3).

Scope boundary:

  * Does NOT hold shard bytes. Responders (storage providers) compute
    their own Merkle trees from on-disk shard content.
  * Does NOT submit on-chain transactions itself. Escalation is
    delegated to `OnChainSlashHook.submit_proof_failure` — the real
    implementation at Foundation-ops time wraps web3 submission to
    `StorageSlashing.submitProofFailure`.
  * Does NOT manage heartbeats. That flow lives in the provider's
    on-chain heartbeat emission (StorageSlashing.recordHeartbeat)
    and a monitoring service that calls
    StorageSlashing.slashForMissingHeartbeat past the grace window.
    Included here only as a protocol-level data class for completeness.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Protocol


logger = logging.getLogger(__name__)


__all__ = [
    "ChallengeIssuer",
    "MerkleTree",
    "OnChainSlashHook",
    "ProofChallenge",
    "ProofResponse",
    "ProofResult",
    "ProofVerdict",
    "ProofVerifier",
    "verify_merkle_proof",
]


# -----------------------------------------------------------------------------
# Merkle tree
# -----------------------------------------------------------------------------


class MerkleTree:
    """Simple SHA-256 Merkle tree over a list of fixed-size chunks.

    Pads the leaf count to the next power of two with zero-digest leaves
    so internal-node hashing is always over two children. Proof is the
    sibling-hash path from leaf up to the root.
    """

    _EMPTY_LEAF = b"\x00" * 32

    def __init__(self, chunks: List[bytes]) -> None:
        if not chunks:
            raise ValueError("MerkleTree requires at least one chunk")
        # Leaf digests.
        leaves = [hashlib.sha256(c).digest() for c in chunks]
        # Pad to a power of two.
        target = 1
        while target < len(leaves):
            target *= 2
        leaves += [self._EMPTY_LEAF] * (target - len(leaves))
        self._levels: List[List[bytes]] = [leaves]
        # Build up.
        while len(self._levels[-1]) > 1:
            prev = self._levels[-1]
            level: List[bytes] = []
            for i in range(0, len(prev), 2):
                level.append(hashlib.sha256(prev[i] + prev[i + 1]).digest())
            self._levels.append(level)
        self._leaf_count = target

    @property
    def leaf_count(self) -> int:
        return self._leaf_count

    def root(self) -> bytes:
        return self._levels[-1][0]

    def proof(self, leaf_index: int) -> List[bytes]:
        if leaf_index < 0 or leaf_index >= self._leaf_count:
            raise IndexError(
                f"leaf_index {leaf_index} out of range [0, {self._leaf_count})"
            )
        path: List[bytes] = []
        idx = leaf_index
        for level in self._levels[:-1]:  # exclude root
            sibling = idx ^ 1  # flip low bit
            path.append(level[sibling])
            idx //= 2
        return path


def verify_merkle_proof(
    leaf: bytes,
    proof: List[bytes],
    leaf_index: int,
    root: bytes,
) -> bool:
    """Verify a Merkle inclusion proof.

    `leaf` is the raw chunk bytes (this function hashes it into a leaf
    digest). Returns True iff the path reconstructs to `root`.
    """
    digest = hashlib.sha256(leaf).digest()
    idx = leaf_index
    for sibling in proof:
        if idx % 2 == 0:
            digest = hashlib.sha256(digest + sibling).digest()
        else:
            digest = hashlib.sha256(sibling + digest).digest()
        idx //= 2
    return digest == root


# -----------------------------------------------------------------------------
# Protocol records
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofChallenge:
    challenge_id: str
    shard_id: str
    provider_id: str
    chunk_index: int      # which chunk the responder must supply
    issued_at_unix: int
    deadline_unix: int
    nonce: bytes          # replay protection


@dataclass(frozen=True)
class ProofResponse:
    challenge_id: str
    chunk_data: bytes           # the raw chunk bytes
    merkle_proof: List[bytes]   # sibling path from leaf to root


class ProofVerdict(str, Enum):
    OK = "ok"
    MISSING_RESPONSE = "missing_response"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CHALLENGE_ID_MISMATCH = "challenge_id_mismatch"
    MERKLE_MISMATCH = "merkle_mismatch"


@dataclass(frozen=True)
class ProofResult:
    challenge: ProofChallenge
    response: Optional[ProofResponse]
    verdict: ProofVerdict

    @property
    def verified(self) -> bool:
        return self.verdict is ProofVerdict.OK


# -----------------------------------------------------------------------------
# On-chain hook
# -----------------------------------------------------------------------------


class OnChainSlashHook(Protocol):
    """Bridge to StorageSlashing.submitProofFailure.

    Implementations wrap web3.py at Foundation-ops time; tests inject a
    stub.
    """

    def submit_proof_failure(
        self,
        provider_id: str,
        shard_id: str,
        evidence_hash: str,
        challenger: str,
    ) -> None: ...


# -----------------------------------------------------------------------------
# Issuer
# -----------------------------------------------------------------------------


class ChallengeIssuer:
    """Issues challenges against storage providers.

    Chunk selection is uniform over [0, num_chunks). Nonce is 16 random
    bytes. Challenge IDs are derived from keccak-equivalent
    sha256(shard_id || provider_id || nonce || issued_at) so the ID is
    verifiable by the responder without a shared clock.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        rng: Optional[Callable[[int], bytes]] = None,
    ) -> None:
        self._clock = clock
        self._rng = rng or secrets.token_bytes

    def issue(
        self,
        provider_id: str,
        shard_id: str,
        num_chunks: int,
        *,
        deadline_seconds: int = 30,
    ) -> ProofChallenge:
        if num_chunks <= 0:
            raise ValueError("num_chunks must be > 0")
        # Uniform random chunk index.
        raw = int.from_bytes(self._rng(4), "big")
        chunk_index = raw % num_chunks
        issued_at = int(self._clock())
        nonce = self._rng(16)
        challenge_id = hashlib.sha256(
            shard_id.encode()
            + provider_id.encode()
            + nonce
            + issued_at.to_bytes(8, "big")
        ).hexdigest()
        return ProofChallenge(
            challenge_id=challenge_id,
            shard_id=shard_id,
            provider_id=provider_id,
            chunk_index=chunk_index,
            issued_at_unix=issued_at,
            deadline_unix=issued_at + deadline_seconds,
            nonce=nonce,
        )


# -----------------------------------------------------------------------------
# Verifier
# -----------------------------------------------------------------------------


class ProofVerifier:
    """Verifies proof responses and escalates failures on-chain.

    `expected_root` for each verification comes from the caller's local
    `ShardManifest` record (committed at upload time). This module is
    oblivious to manifest storage — the caller fetches the root.

    On failure, if `slash_hook` is configured, the verifier calls
    `slash_hook.submit_proof_failure` with an `evidence_hash` derived
    from the (challenge_id, response_chunk, response_proof, verdict)
    tuple — auditable on-chain, opaque to the contract.
    """

    def __init__(
        self,
        *,
        challenger_id: str,
        clock: Callable[[], float] = time.time,
        slash_hook: Optional[OnChainSlashHook] = None,
    ) -> None:
        self._challenger_id = challenger_id
        self._clock = clock
        self._slash_hook = slash_hook

    def verify(
        self,
        challenge: ProofChallenge,
        response: Optional[ProofResponse],
        expected_root: bytes,
    ) -> ProofResult:
        verdict = self._classify(challenge, response, expected_root)
        result = ProofResult(
            challenge=challenge, response=response, verdict=verdict
        )
        if not result.verified and self._slash_hook is not None:
            self._escalate(result)
        return result

    def _classify(
        self,
        challenge: ProofChallenge,
        response: Optional[ProofResponse],
        expected_root: bytes,
    ) -> ProofVerdict:
        if response is None:
            # Distinguish deadline-exceeded from missing-response where
            # possible — missing past the deadline is a deadline
            # violation, missing before the deadline is not yet
            # considered a violation for this verifier (caller chose to
            # give up early).
            now = int(self._clock())
            if now > challenge.deadline_unix:
                return ProofVerdict.DEADLINE_EXCEEDED
            return ProofVerdict.MISSING_RESPONSE

        # Past-deadline response still counts as DEADLINE_EXCEEDED — the
        # challenger cannot retroactively accept a late submission.
        if int(self._clock()) > challenge.deadline_unix:
            return ProofVerdict.DEADLINE_EXCEEDED

        if response.challenge_id != challenge.challenge_id:
            return ProofVerdict.CHALLENGE_ID_MISMATCH

        if not verify_merkle_proof(
            response.chunk_data,
            response.merkle_proof,
            challenge.chunk_index,
            expected_root,
        ):
            return ProofVerdict.MERKLE_MISMATCH

        return ProofVerdict.OK

    def _escalate(self, result: ProofResult) -> None:
        assert self._slash_hook is not None
        challenge = result.challenge
        evidence_blob = (
            challenge.challenge_id
            + ":"
            + result.verdict.value
        ).encode()
        if result.response is not None:
            evidence_blob += b":" + hashlib.sha256(
                result.response.chunk_data
            ).digest()
        evidence_hash = "0x" + hashlib.sha256(evidence_blob).hexdigest()
        try:
            self._slash_hook.submit_proof_failure(
                provider_id=challenge.provider_id,
                shard_id=challenge.shard_id,
                evidence_hash=evidence_hash,
                challenger=self._challenger_id,
            )
        except Exception as exc:  # noqa: BLE001
            # sp1000 — make a NON-LANDING slash loud + actionable instead of a
            # generic swallowed error. In particular the CallerNotSlasher revert
            # is the known on-chain wiring gap: StorageSlashing is NOT StakeBond's
            # authorized slasher (StakeBond's immutable slasher is the
            # BatchSettlementRegistry), so EVERY storage proof-failure slash
            # reverts and the misbehaving provider is never penalized — the
            # storage-tier stake deterrent is non-functional until fixed on-chain.
            _m = str(exc).lower()
            if "callernotslasher" in _m or "not slasher" in _m or "slasher" in _m:
                logger.critical(
                    "Storage-slash for provider %s did NOT land (%s): "
                    "StorageSlashing is not StakeBond's authorized slasher "
                    "(on-chain wiring gap — StakeBond's immutable slasher is the "
                    "BatchSettlementRegistry). The storage-tier stake deterrent is "
                    "NON-FUNCTIONAL until fixed on-chain (see "
                    "docs/2026-06-04-storage-slashing-slasher-wiring-gap.md). The "
                    "misbehaving provider was NOT penalized for challenge %s.",
                    challenge.provider_id, type(exc).__name__,
                    challenge.challenge_id,
                )
            else:
                logger.exception(
                    "slash-hook submission failed for challenge %s — provider %s "
                    "was NOT penalized (not auto-retried)",
                    challenge.challenge_id, challenge.provider_id,
                )
