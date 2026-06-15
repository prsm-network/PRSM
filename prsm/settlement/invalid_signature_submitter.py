"""On-chain broadcast of an INVALID_SIGNATURE challenge (challenge/dispute brick 4).

Brick 3 (challenge_assembler) produces an ``InvalidSignatureChallenge`` (leaf + merkle
proof + auxData). This module broadcasts it to ``BatchSettlementRegistry.challengeReceipt``
to invalidate the fraudulent receipt's value and slash the provider's bond.

It is INERT by default — broadcasting is a USER-GATED action (it spends gas and slashes a
provider's staked bond; per the standing rule the assistant assembles + verifies
read-only, the user signs). Two entry points:
  - ``dry_run(challenge)``: READ-ONLY static call (eth_call) — simulates the challenge
    against current chain state and returns whether it WOULD succeed, WITHOUT broadcasting
    or spending gas. This is the assistant/auditor's pre-flight check.
  - ``submit(challenge)``: signs + broadcasts the real transaction. Requires a funded
    challenger key. Never raises — returns a uniform ChallengeResult.

Mirrors marketplace/consensus_submitter.py (the CONSENSUS_MISMATCH sibling): same ABI,
nonce-lock, gas floor, and three-tier broadcast/pending/reverted error classification.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)
from prsm.settlement.challenge_assembler import (
    REASON_INVALID_SIGNATURE,
    InvalidSignatureChallenge,
)

logger = logging.getLogger(__name__)

# Same floor as the consensus submitter — the registry requires
# gasleft() >= MIN_SLASH_GAS before the slash try/catch; 1M leaves headroom
# for Merkle verify + binding checks + slash + event emission.
DEFAULT_CHALLENGE_GAS = 1_000_000

# The challengeReceipt ABI (identical to consensus_submitter's — one registry function).
CHALLENGE_RECEIPT_ABI = [{
    "inputs": [
        {"name": "batchId", "type": "bytes32"},
        {
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


@dataclass(frozen=True)
class ChallengeResult:
    """Outcome of a broadcast attempt. Mirrors consensus_submitter.ChallengeResult."""
    success: bool
    tx_hash_hex: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]


@dataclass(frozen=True)
class DryRunResult:
    """Read-only pre-flight verdict — would the challenge succeed if broadcast?"""
    would_succeed: bool
    revert_reason: Optional[str] = None


class InvalidSignatureChallengeSubmitter:
    """Sync Web3 client for INVALID_SIGNATURE challenges. One instance per challenger
    keypair (the one-keypair-per-process invariant). ``registry``/``account``/``web3`` can
    be injected for tests; otherwise built from rpc_url/registry_address/private_key."""

    def __init__(
        self,
        *,
        rpc_url: Optional[str] = None,
        registry_address: Optional[str] = None,
        private_key: Optional[str] = None,
        gas_budget: int = DEFAULT_CHALLENGE_GAS,
        web3: Any = None,
        registry: Any = None,
        account: Any = None,
    ) -> None:
        self.gas_budget = gas_budget
        self._tx_lock = threading.Lock()
        # Injected (test) path.
        if registry is not None and account is not None:
            self.web3 = web3
            self.registry = registry
            self._account = account
            return
        # Live path — requires web3 + a funded challenger key.
        if not HAS_WEB3:
            raise RuntimeError(
                "web3 package is required (pip install web3 eth-account)"
            )
        if not (rpc_url and registry_address and private_key):
            raise ValueError(
                "rpc_url, registry_address and private_key are required for the live "
                "path (or inject registry+account for tests)"
            )
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.registry = self.web3.eth.contract(
            address=Web3.to_checksum_address(registry_address),
            abi=CHALLENGE_RECEIPT_ABI,
        )
        self._account = Account.from_key(private_key)

    @property
    def address(self) -> str:
        return self._account.address

    # ── read-only pre-flight (NOT a broadcast) ─────────────────────────────────────

    def dry_run(self, challenge: InvalidSignatureChallenge) -> DryRunResult:
        """Static eth_call of challengeReceipt against current chain state — no tx, no
        gas, no state change. Returns whether the challenge WOULD succeed. The
        assistant/auditor runs this before the user broadcasts."""
        if int(challenge.reason_code) != REASON_INVALID_SIGNATURE:
            return DryRunResult(
                would_succeed=False,
                revert_reason=f"not an INVALID_SIGNATURE challenge "
                              f"(reason={challenge.reason_code})",
            )
        try:
            self.registry.functions.challengeReceipt(
                *challenge.to_call_args()
            ).call({"from": self.address})
            return DryRunResult(would_succeed=True)
        except Exception as exc:  # noqa: BLE001 — surface the revert reason, don't raise
            return DryRunResult(would_succeed=False, revert_reason=str(exc))

    # ── broadcast (USER-GATED) ─────────────────────────────────────────────────────

    def submit(self, challenge: InvalidSignatureChallenge) -> ChallengeResult:
        """Sign + broadcast the challengeReceipt transaction. USER-GATED (spends gas +
        slashes a provider bond). Never raises — returns a uniform ChallengeResult."""
        if int(challenge.reason_code) != REASON_INVALID_SIGNATURE:
            return ChallengeResult(
                success=False, tx_hash_hex=None,
                error_type="ValueError",
                error_message=f"reason_code {challenge.reason_code} is not "
                              f"INVALID_SIGNATURE ({REASON_INVALID_SIGNATURE})",
            )
        try:
            with self._tx_lock:
                tx = self.registry.functions.challengeReceipt(
                    *challenge.to_call_args()
                ).build_transaction(self._tx_overrides())
                tx_hash_hex, _ = self._sign_and_send(tx)
            return ChallengeResult(
                success=True, tx_hash_hex=tx_hash_hex,
                error_type=None, error_message=None,
            )
        except (BroadcastFailedError, OnChainPendingError,
                OnChainRevertedError) as exc:
            logger.warning(
                "invalid-signature challenge failed: %s: %s",
                type(exc).__name__, exc,
            )
            return ChallengeResult(
                success=False,
                tx_hash_hex=getattr(exc, "tx_hash", None),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 — uniform result; log loudly
            logger.error(
                "invalid-signature challenge raised unexpected %s: %s",
                type(exc).__name__, exc,
            )
            return ChallengeResult(
                success=False, tx_hash_hex=None,
                error_type=type(exc).__name__, error_message=str(exc),
            )

    # ── internals ──────────────────────────────────────────────────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending",
            ),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
            "gas": self.gas_budget,
        }

    def _sign_and_send(self, tx: dict) -> Tuple[str, "TransferStatus"]:
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None,
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
                f"broadcast OK but receipt unknown: {exc}", tx_hash=tx_hash_hex,
            ) from exc
        if receipt.status != 1:
            raise OnChainRevertedError(f"challenge tx reverted: {tx_hash_hex}")
        return tx_hash_hex, TransferStatus.CONFIRMED
