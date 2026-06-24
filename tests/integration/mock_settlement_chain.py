"""Phase 3.1 Task 8 — Python-only mock of the on-chain settlement surface.

Simulates the BatchSettlementRegistry + EscrowPool contracts in-process
for end-to-end integration testing. Matches the SettlementContractClient
protocol exactly (Task 6), so BatchSettlementClient can drive this mock
without knowing it isn't a real chain.

What the mock simulates:
  - Deterministic batch_id derivation matching the Solidity logic
  - PENDING → FINALIZED state machine with a challenge-window clock
  - EscrowPool-style per-requester balances
  - FTNS transfer from requester balance to provider on finalize

What it explicitly does NOT simulate (scope discipline):
  - Ed25519 signature verification on challenges
  - Merkle-proof verification on challenges (Task 3 contract logic)
  - Real gas costs

Real on-chain integration (hardhat-run against the actual Solidity
contracts) is Task 10 (post-hardware). This mock validates the Python-
side pipeline end-to-end.
"""
from __future__ import annotations

import asyncio
import hashlib
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class _MockBatch:
    batch_id: bytes
    provider: str
    requester: str
    merkle_root: bytes
    receipt_count: int
    total_value_ftns: int
    commit_timestamp: int
    status: int = 1  # 1=PENDING, 2=FINALIZED


class MockSettlementChain:
    """In-process replacement for BatchSettlementRegistry + EscrowPool.

    Designed for E2E integration tests: BatchSettlementClient talks to
    instances of this class (via SettlementContractClient protocol) and
    observes realistic state transitions without any blockchain.

    Time control: tests call `advance_time(seconds)` to simulate
    challenge-window elapsing, decoupling the test from wall clock.
    """

    def __init__(self, challenge_window_seconds: int = 3 * 24 * 3600):
        self._batches: Dict[bytes, _MockBatch] = {}
        # Per-requester FTNS balances (in wei — same units as uint128).
        self._balances: Dict[str, int] = {}
        self._challenge_window = challenge_window_seconds
        self._provider_batch_sequence: Dict[str, int] = {}
        self._simulated_now: int = int(_time.time())
        # Per-node clients created via `for_provider` — keyed by
        # provider address so they can be resolved for tests.
        self._clients: Dict[str, "MockContractClient"] = {}

    # ── Time control (test-only) ───────────────────────────────

    @property
    def simulated_now(self) -> int:
        return self._simulated_now

    def advance_time(self, seconds: int) -> None:
        """Move the simulated clock forward. Does NOT touch any batches
        directly — the state check happens at read time (isFinalizable)."""
        self._simulated_now += seconds

    # ── EscrowPool surface ─────────────────────────────────────

    def deposit(self, requester: str, amount: int) -> None:
        """Simulated EscrowPool.deposit — requester wallet → pool balance."""
        self._balances[requester] = self._balances.get(requester, 0) + amount

    def balance_of(self, addr: str) -> int:
        return self._balances.get(addr, 0)

    # ── BatchSettlementRegistry surface ────────────────────────

    def _derive_batch_id(
        self,
        provider: str,
        requester: str,
        merkle_root: bytes,
        receipt_count: int,
    ) -> bytes:
        """Deterministic batch_id derivation matching the Solidity logic
        in BatchSettlementRegistry.commitBatch: keccak256(abi.encode(
        provider, requester, merkleRoot, receiptCount, block.number,
        sequence))."""
        sequence = self._provider_batch_sequence.get(provider, 0)
        self._provider_batch_sequence[provider] = sequence + 1
        preimage = (
            provider.encode()
            + requester.encode()
            + merkle_root
            + receipt_count.to_bytes(32, "big")
            + (self._simulated_now).to_bytes(32, "big")  # stand-in for block.number
            + sequence.to_bytes(32, "big")
        )
        return hashlib.sha256(preimage).digest()

    async def commit_batch(
        self,
        provider: str,
        requester: str,
        merkle_root: bytes,
        receipt_count: int,
        total_value_ftns: int,
        metadata_uri: str = "",
        *,
        tier_slash_rate_bps: int = 0,
        consensus_group_id: bytes = b"\x00" * 32,
        attestation_commitment: bytes = b"",  # sp1242 — accepted, mock ignores
    ) -> Tuple[bytes, int]:
        """Mock of BatchSettlementRegistry.commitBatch. Returns
        (batch_id, commit_timestamp_unix).

        tier_slash_rate_bps (Phase 7) + consensus_group_id (Phase 7.1x)
        accepted for wire compatibility with the real
        SettlementContractClient.commit_batch signature. Not used by the
        mock — they're informational fields that the real contract
        snapshots for slashing / consensus-group attribution."""
        if receipt_count == 0:
            raise ValueError("mock chain: zero receiptCount")
        if not merkle_root or merkle_root == b"\x00" * 32:
            raise ValueError("mock chain: empty merkle root")

        batch_id = self._derive_batch_id(
            provider, requester, merkle_root, receipt_count
        )
        self._batches[batch_id] = _MockBatch(
            batch_id=batch_id,
            provider=provider,
            requester=requester,
            merkle_root=merkle_root,
            receipt_count=receipt_count,
            total_value_ftns=total_value_ftns,
            commit_timestamp=self._simulated_now,
            status=1,
        )
        return batch_id, self._simulated_now

    async def is_finalizable(self, batch_id: bytes) -> bool:
        b = self._batches.get(batch_id)
        if b is None or b.status != 1:
            return False
        elapsed = self._simulated_now - b.commit_timestamp
        return elapsed >= self._challenge_window

    async def finalize_batch(self, batch_id: bytes) -> None:
        b = self._batches.get(batch_id)
        if b is None:
            raise ValueError(f"mock chain: unknown batch_id {batch_id.hex()[:12]}…")
        if b.status != 1:
            raise ValueError(
                f"mock chain: batch not PENDING (status={b.status})"
            )
        elapsed = self._simulated_now - b.commit_timestamp
        if elapsed < self._challenge_window:
            raise ValueError(
                f"mock chain: challenge window not elapsed "
                f"({elapsed}s < {self._challenge_window}s)"
            )
        # Transfer FTNS requester → provider. Simulates EscrowPool.settleFromRequester.
        avail = self._balances.get(b.requester, 0)
        if avail < b.total_value_ftns:
            raise ValueError(
                f"mock chain: requester {b.requester[:12]}… has "
                f"{avail} wei, need {b.total_value_ftns}"
            )
        self._balances[b.requester] = avail - b.total_value_ftns
        self._balances[b.provider] = (
            self._balances.get(b.provider, 0) + b.total_value_ftns
        )
        b.status = 2  # FINALIZED

    async def get_batch_status(self, batch_id: bytes) -> int:
        b = self._batches.get(batch_id)
        return b.status if b is not None else 0

    # ── Provider-scoped client factory ─────────────────────────

    def for_provider(self, provider_address: str) -> "MockContractClient":
        """Return a SettlementContractClient bound to a specific provider
        address. BatchSettlementClient uses one of these per provider."""
        client = self._clients.get(provider_address)
        if client is None:
            client = MockContractClient(self, provider_address)
            self._clients[provider_address] = client
        return client


class MockContractClient:
    """SettlementContractClient implementation backed by MockSettlementChain.

    Satisfies the Task 6 protocol exactly. Binds to one provider address
    (since commit_batch needs to know the provider's identity)."""

    def __init__(self, chain: MockSettlementChain, provider_address: str):
        self._chain = chain
        self._provider = provider_address

    async def commit_batch(
        self,
        provider_address: str,
        requester_address: str,
        merkle_root: bytes,
        receipt_count: int,
        total_value_ftns: int,
        tier_slash_rate_bps: int = 0,
        consensus_group_id: bytes = b"\x00" * 32,
        metadata_uri: str = "",
        attestation_commitment: bytes = b"",  # sp1242 — accepted, mock ignores
    ) -> Tuple[bytes, int]:
        """Mirrors SettlementContractClient.commit_batch exactly —
        positional args + tier_slash_rate_bps (Phase 7) +
        consensus_group_id (Phase 7.1x). Forwards everything to
        MockSettlementChain.commit_batch."""
        return await self._chain.commit_batch(
            provider=provider_address,
            requester=requester_address,
            merkle_root=merkle_root,
            receipt_count=receipt_count,
            total_value_ftns=total_value_ftns,
            metadata_uri=metadata_uri,
            tier_slash_rate_bps=tier_slash_rate_bps,
            consensus_group_id=consensus_group_id,
        )

    async def is_finalizable(self, batch_id: bytes) -> bool:
        return await self._chain.is_finalizable(batch_id)

    async def finalize_batch(self, batch_id: bytes) -> None:
        await self._chain.finalize_batch(batch_id)

    async def get_batch_status(self, batch_id: bytes) -> int:
        return await self._chain.get_batch_status(batch_id)
