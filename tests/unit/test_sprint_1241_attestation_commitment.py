"""Sprint 1241 (TEE Tier-3 roadmap F off-chain activation) — batch attestation
commitment + client routing to commitBatchWithAttestation.

batch_attestation_commitment() turns a batch's per-receipt tee_attestation audit
dicts into the deterministic bytes32 the on-chain commitBatchWithAttestation
stores (weakest-link tier + measurement). The settlement client routes to the new
contract function only when a non-zero commitment is supplied AND the deployed
registry supports it (else legacy commitBatch — so un-upgraded deployments never
get a failed tx). Default OFF.
"""

import json
from unittest.mock import MagicMock, patch

from eth_abi import encode as abi_encode
from eth_utils import keccak

from prsm.compute.shard_receipt import (
    attestation_tier_from_audit_dict,
    batch_attestation_commitment,
    tee_attestation_audit_dict,
)
from prsm.compute.tee.models import TEEType
from prsm.enterprise.tee_policy import AttestationTier, tier_rank


def _audit(tee_type, *, is_dev_only=False, is_multi_stage=True):
    return {"attestation_b64": "x", "is_dev_only": is_dev_only,
            "is_multi_stage": is_multi_stage, "tee_type": tee_type}


# ── tier mapping from the audit dict ────────────────────────────────

def test_tier_mapping():
    assert attestation_tier_from_audit_dict(None) == AttestationTier.NONE
    assert attestation_tier_from_audit_dict({}) == AttestationTier.NONE
    assert attestation_tier_from_audit_dict(_audit("none")) == AttestationTier.NONE
    assert attestation_tier_from_audit_dict(_audit("software")) == AttestationTier.SOFTWARE
    # dev-only is a software stub regardless of claimed type
    assert attestation_tier_from_audit_dict(_audit("sgx", is_dev_only=True)) == AttestationTier.SOFTWARE
    # a real hardware quote maps to UNVERIFIED (audit dict has no vendor_verified)
    assert attestation_tier_from_audit_dict(_audit("sgx")) == AttestationTier.HARDWARE_UNVERIFIED
    assert attestation_tier_from_audit_dict(_audit("sev")) == AttestationTier.HARDWARE_UNVERIFIED


# ── commitment: empty / all-none → zero ─────────────────────────────

def test_commitment_zero_when_no_attestation():
    assert batch_attestation_commitment([]) == b"\x00" * 32
    assert batch_attestation_commitment([None, None]) == b"\x00" * 32
    assert batch_attestation_commitment(None) == b"\x00" * 32


# ── commitment: deterministic + auditor-reproducible ────────────────

def test_commitment_deterministic_and_reproducible():
    dicts = [_audit("sgx"), _audit("sgx")]
    c1 = batch_attestation_commitment(dicts)
    c2 = batch_attestation_commitment(list(reversed(dicts)))  # order-independent (sorted)
    assert c1 == c2 and len(c1) == 32 and c1 != b"\x00" * 32

    # independently reproduce the documented formula
    triples = sorted([tier_rank(AttestationTier.HARDWARE_UNVERIFIED), "sgx", True]
                     for _ in dicts)
    measurement = keccak(json.dumps(triples, separators=(",", ":")).encode())
    expected = keccak(abi_encode(
        ["uint8", "bytes32", "bool"],
        [tier_rank(AttestationTier.HARDWARE_UNVERIFIED), measurement, True]))
    assert c1 == expected


# ── commitment: weakest-link ────────────────────────────────────────

def test_commitment_weakest_link():
    all_hw = batch_attestation_commitment([_audit("sgx"), _audit("sgx")])
    one_soft = batch_attestation_commitment([_audit("sgx"), _audit("software")])
    assert all_hw != one_soft  # one software stage collapses the batch tier


def test_commitment_reflects_all_multi_stage_flag():
    a = batch_attestation_commitment([_audit("sgx", is_multi_stage=True)])
    b = batch_attestation_commitment([_audit("sgx", is_multi_stage=False)])
    assert a != b


def test_commitment_from_real_audit_dicts():
    # end-to-end with the sp1238 helper producing the dicts
    d = tee_attestation_audit_dict(b"PRSM-MS-ATT-V1:" + b"\x01" * 16, TEEType.SGX)
    c = batch_attestation_commitment([d])
    assert len(c) == 32 and c != b"\x00" * 32


# ── client routing ──────────────────────────────────────────────────

def _make_client(supports_attestation):
    from prsm.economy.web3.batch_settlement_contract_client import (
        Web3SettlementContractClient,
    )
    c = Web3SettlementContractClient(
        rpc_url="http://localhost:8545",
        contract_address="0x" + "a" * 40,
        private_key="0x" + "1" * 64,
        supports_attestation=supports_attestation,
    )
    # mock the on-chain pieces
    c.contract = MagicMock()
    c._sign_send_wait = MagicMock(return_value="receipt")
    c.contract.events.BatchCommitted().process_receipt.return_value = [
        {"args": {"batchId": b"\x07" * 32, "commitTimestamp": 1700000000}}]
    c._tx_overrides = MagicMock(return_value={})
    return c


ARGS = ("0x" + "b" * 40, b"\x02" * 32, 10, 1_000, 0, b"\x00" * 32, "")
ATT = b"\x09" * 32


def test_routes_to_attestation_fn_when_supported_and_nonzero():
    c = _make_client(supports_attestation=True)
    c._commit_batch_sync(*ARGS, attestation_commitment=ATT)
    c.contract.functions.commitBatchWithAttestation.assert_called_once()
    c.contract.functions.commitBatch.assert_not_called()


def test_routes_to_legacy_when_not_supported():
    c = _make_client(supports_attestation=False)
    c._commit_batch_sync(*ARGS, attestation_commitment=ATT)  # commitment ignored
    c.contract.functions.commitBatch.assert_called_once()
    c.contract.functions.commitBatchWithAttestation.assert_not_called()


def test_routes_to_legacy_when_commitment_zero():
    c = _make_client(supports_attestation=True)
    c._commit_batch_sync(*ARGS, attestation_commitment=b"\x00" * 32)
    c.contract.functions.commitBatch.assert_called_once()
    c.contract.functions.commitBatchWithAttestation.assert_not_called()


def test_default_no_attestation_uses_legacy():
    c = _make_client(supports_attestation=True)
    c._commit_batch_sync(*ARGS)  # no attestation_commitment
    c.contract.functions.commitBatch.assert_called_once()
    c.contract.functions.commitBatchWithAttestation.assert_not_called()
