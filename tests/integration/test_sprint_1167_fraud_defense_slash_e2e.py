"""Sprint 1167 — LOCAL HARDHAT E2E for the settlement fraud-defense OPERABILITY LOOP.

The fraud-defense operability arc (sp1149→1166) is unit-tested + adversarially verified, but
the prepare→broadcast path had never been exercised against a REAL EVM end-to-end. sp1165's
verification caught a latent leaf-hash CONVENTION bug (utf8 vs b64decode) that unit tests
missed because no test cross-checked the assembled challenge against an actual committed root +
the on-chain verifier. This bench is that proof: it boots a real Hardhat node, deploys the full
settlement suite WITH THE REAL Ed25519Verifier, commits a batch containing a GENUINELY-INVALID
signature, then drives the ACTUAL operator path —

    AuditFindingRecord -> prepare_challenge_from_finding (the sp1163 bridge)
                       -> gated_broadcast_prepared (the sp1166 broadcaster)
                       -> ChallengeSubmitter live (dry_run + submit on the real chain)

— and asserts the on-chain ``challengeReceipt`` ACCEPTS the challenge (the merkle proof verifies
against the committed root AND the real Ed25519Verifier confirms the signature is invalid) and
INVALIDATES the fraudulent receipt (``invalidatedValueFTNS`` rises by the leaf value).

This validates the WHOLE leaf/merkle/auxData convention + the verifier wiring + the bridge +
the broadcaster against a real EVM — the strongest possible proof, and exactly the class of test
that would have caught the consensus convention bug. A genuine NEGATIVE control (an HONEST
receipt) confirms fraud-specificity: the assembler refuses to even build a challenge for it.

Slashing the provider's bond is a best-effort, stake-bond-gated step (BatchSettlementRegistry.sol
:698-735); with no bond configured (tier_slash_rate_bps==0) the challenge still PROVES the fraud
and invalidates the receipt — which is exactly the on-chain-agreement property under test here.

Skipped automatically if ``npx hardhat`` is not on PATH / node_modules absent.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

from subprocess import Popen as _REAL_POPEN  # noqa: E402 — capture real Popen pre-conftest-patch

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"

_ONE_FTNS = 10 ** 18


def _hardhat_available() -> bool:
    return (
        shutil.which("npx") is not None
        and (CONTRACTS_DIR / "node_modules" / ".bin" / "hardhat").exists()
    )


requires_hardhat = pytest.mark.skipif(
    not _hardhat_available(), reason="Hardhat / npx not available"
)


@pytest.fixture(scope="module")
def hardhat_node():
    """Local hardhat node on :8545 (== hardhat's built-in ``localhost`` network URL, so the
    inline deploy can use ``--network localhost``). Module-scoped + serial (do not xdist
    against the other :8545 benches)."""
    log_path = Path("/tmp/prsm_sp1167_fraud_hardhat.log")
    log_fh = open(log_path, "wb")
    proc = _REAL_POPEN(
        "exec npx hardhat node --port 8545",
        cwd=str(CONTRACTS_DIR), stdout=log_fh, stderr=subprocess.STDOUT,
        shell=True, start_new_session=True,
    )
    try:
        deadline = time.time() + 120.0
        port_open = False
        while time.time() < deadline:
            if proc.poll() is not None:
                log_fh.close()
                raise RuntimeError(
                    f"hardhat node exited early (rc={proc.returncode})\n"
                    f"{log_path.read_text(errors='replace')[-2000:]}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                try:
                    s.connect(("127.0.0.1", 8545))
                    port_open = True
                    break
                except OSError:
                    time.sleep(0.5)
        if not port_open:
            log_fh.close()
            raise TimeoutError(
                f"hardhat node did not open :8545 within 120s\n"
                f"{log_path.read_text(errors='replace')[-2000:]}")
        time.sleep(1.0)
        yield "http://127.0.0.1:8545"
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        finally:
            try:
                log_fh.close()
            except Exception:
                pass


@pytest.fixture(scope="module")
def deployed(hardhat_node):
    """Deploy Token -> Registry (1-hour window) -> Pool -> setEscrowPool -> the REAL
    Ed25519Verifier -> setSignatureVerifier. The REAL verifier (not the mock) is the whole
    point: a genuinely-invalid signature must verify FALSE on-chain so the INVALID_SIGNATURE
    challenge is PROVEN. No StakeBond (slash is best-effort + not under test)."""
    deploy_js = """
const hre = require("hardhat");
const fs = require("fs");
async function main() {
  const [deployer, requester, node0, challenger] = await hre.ethers.getSigners();
  const Token = await hre.ethers.getContractFactory("MockERC20");
  const token = await Token.deploy();
  await token.waitForDeployment();
  const Registry = await hre.ethers.getContractFactory("BatchSettlementRegistry");
  const registry = await Registry.deploy(deployer.address, 3600);
  await registry.waitForDeployment();
  const Pool = await hre.ethers.getContractFactory("EscrowPool");
  const pool = await Pool.deploy(
    deployer.address, await token.getAddress(), await registry.getAddress());
  await pool.waitForDeployment();
  await registry.connect(deployer).setEscrowPool(await pool.getAddress());
  // REAL Ed25519 verifier (a genuinely-invalid sig must verify FALSE on-chain).
  const Verifier = await hre.ethers.getContractFactory("Ed25519Verifier");
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();
  await registry.connect(deployer).setSignatureVerifier(await verifier.getAddress());
  const out = {
    token: await token.getAddress(), pool: await pool.getAddress(),
    registry: await registry.getAddress(), verifier: await verifier.getAddress(),
    requester: requester.address, node0: node0.address, challenger: challenger.address,
    node0Key: "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
    challengerKey: "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
  };
  fs.writeFileSync("/tmp/prsm_sp1167_fraud_deploy.json", JSON.stringify(out));
}
main().catch((e) => { console.error(e); process.exit(1); });
"""
    script = CONTRACTS_DIR / "scripts" / "_sp1167_fraud_deploy.js"
    script.write_text(deploy_js)

    def _run(argv):
        p = _REAL_POPEN(argv, cwd=str(CONTRACTS_DIR))
        if p.wait() != 0:
            raise RuntimeError(f"command failed: {' '.join(argv)}")

    try:
        _run(["npx", "hardhat", "compile"])
        _run(["npx", "hardhat", "run", "scripts/_sp1167_fraud_deploy.js",
              "--network", "localhost"])
        with open("/tmp/prsm_sp1167_fraud_deploy.json") as f:
            return json.load(f)
    finally:
        if script.exists():
            script.unlink()


def _batched_receipt(identity, *, requester, provider, job="job-1", shard=0, valid=True,
                     out=None, value=_ONE_FTNS, escrow="e1"):
    """A challenge-defensible BatchedReceipt. valid=False signs a DIFFERENT payload, so the
    committed signature does NOT verify over this leaf's signingMessageHash (genuine fraud)."""
    from prsm.compute.shard_receipt import (
        ShardExecutionReceipt,
        build_receipt_signing_payload,
    )
    from prsm.settlement.accumulator import BatchedReceipt

    out = out or ("ab" * 32)
    if valid:
        sig = identity.sign(build_receipt_signing_payload(
            job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000))
    else:
        sig = identity.sign(build_receipt_signing_payload(
            job_id=job, shard_index=shard, output_hash=("cd" * 32), executed_at_unix=1000))
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address=requester, provider_address=provider,
        value_ftns=value, local_escrow_id=escrow)


@requires_hardhat
def test_invalid_signature_fraud_defense_onchain_e2e(hardhat_node, deployed):
    from web3 import Web3

    from prsm.economy.web3.batch_settlement_contract_client import (
        Web3SettlementContractClient,
    )
    from prsm.node.identity import generate_node_identity
    from prsm.settlement.accumulator import AccumulatorConfig, ReceiptAccumulator
    from prsm.settlement.challenge_broadcaster import gated_broadcast_prepared
    from prsm.settlement.challenge_preparer import prepare_challenge_from_finding
    from prsm.settlement.client import BatchSettlementClient
    from prsm.settlement.invalid_signature_submitter import ChallengeSubmitter
    from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
    from prsm.settlement.settlement_audit_report import AuditFindingRecord

    requester = Web3.to_checksum_address(deployed["requester"])
    provider = Web3.to_checksum_address(deployed["node0"])
    idn = generate_node_identity(display_name="sp1167-provider")

    # A FRAUDULENT receipt: its committed signature signs a DIFFERENT payload, so it does
    # NOT verify over this leaf's signingMessageHash (genuine INVALID_SIGNATURE fraud).
    fraud = _batched_receipt(idn, requester=requester, provider=provider,
                             job="J", shard=1, valid=False, out=("22" * 32))
    receipts = [fraud]  # single-leaf batch: committed root == the fraud leaf hash

    # Commit the batch on the REAL chain (provider commits with its own key). count_threshold=1
    # commits the single receipt immediately (no value-threshold early-trigger ambiguity).
    contract_client = Web3SettlementContractClient(
        rpc_url=hardhat_node, contract_address=deployed["registry"],
        private_key=deployed["node0Key"])
    client = BatchSettlementClient(
        accumulator=ReceiptAccumulator(
            AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600,
                              value_threshold_ftns=10 ** 9)),
        contract_client=contract_client, provider_address=provider)

    async def _commit():
        await client.accumulate(fraud)
        return await client.commit_ready_batches()

    committed = asyncio.run(_commit())
    assert len(committed) == 1, f"expected 1 committed batch, got {len(committed)}"
    batch_id = committed[0].batch_id
    assert committed[0].receipt_count == 1
    fraud_leaf_value = fraud.value_ftns

    # The receipt source the prepare bridge re-sources from (the committed ordered set).
    bid_hex = batch_id.hex()
    receipt_lookup = (
        lambda b: list(receipts) if str(b).lower().removeprefix("0x") == bid_hex else None)

    # NEGATIVE control (fraud-specificity): an HONEST receipt cannot be challenged — the
    # assembler fail-fasts because its committed signature VERIFIES (no fraud to prove). This
    # is a LOCAL refusal (no commit needed) proving the path only challenges genuine fraud.
    honest = _batched_receipt(idn, requester=requester, provider=provider,
                              job="J", shard=0, valid=True, out=("11" * 32))
    honest_record = AuditFindingRecord(
        reason="invalid_signature", batch_id_hex="00" * 32, target_index=0,
        on_chain_reason=1, detail={"receipt_index": 0})
    with pytest.raises(ValueError):
        prepare_challenge_from_finding(
            record=honest_record, receipt_lookup=lambda b: [honest])

    # The OPERATOR PATH for the FRAUDULENT receipt (index 0): finding -> prepare -> broadcast.
    fraud_record = AuditFindingRecord(
        reason="invalid_signature", batch_id_hex=bid_hex, target_index=0,
        on_chain_reason=1, detail={"receipt_index": 0})
    prepared = prepare_challenge_from_finding(
        record=fraud_record, receipt_lookup=receipt_lookup)
    assert prepared.on_chain_reason == 1
    # the prepared leaf is the fraudulent receipt's canonical committed leaf
    assert prepared.challenge.leaf == batched_receipt_to_leaf(fraud)

    submitter = ChallengeSubmitter(
        rpc_url=hardhat_node, registry_address=deployed["registry"],
        private_key=deployed["challengerKey"])

    # Read invalidated value BEFORE (full struct; the client's get_batch omits it).
    def _invalidated_value() -> int:
        return int(contract_client.contract.functions.batches(batch_id).call()[5])

    before = _invalidated_value()

    # ON-CHAIN AGREEMENT cross-checks (the class of bug sp1165's verify caught): the Python
    # committed root == the on-chain root, and the prepared leaf re-hashes to the committed
    # fraud leaf (Python batched_receipt_to_leaf == the contract's _hashLeaf).
    from prsm.settlement.merkle import verify_merkle_proof
    onchain_root = bytes(contract_client.contract.functions.batches(batch_id).call()[2])
    assert onchain_root == committed[0].merkle_root
    assert hash_leaf(prepared.challenge.leaf) == hash_leaf(batched_receipt_to_leaf(fraud))
    assert verify_merkle_proof(
        prepared.challenge.merkle_proof, onchain_root, hash_leaf(prepared.challenge.leaf))

    # USER-GATED broadcast against the REAL chain (dry-run-first inside).
    outcome = gated_broadcast_prepared(
        prepared, submitter=submitter, broadcast_enabled=True)

    # ── on-chain assertions ──
    assert outcome.dry_run_ok is True, (
        f"dry-run should pass on-chain (the real Ed25519Verifier must confirm the sig is "
        f"invalid + the merkle proof must verify): {outcome.dry_run_reason}")
    assert outcome.broadcast is True, (
        f"challenge should broadcast + succeed: {outcome.error_message}")
    assert outcome.tx_hash_hex

    after = _invalidated_value()
    assert after - before == fraud_leaf_value, (
        f"the fraudulent receipt must be invalidated on-chain (invalidatedValueFTNS should "
        f"rise by the leaf value {fraud_leaf_value}); before={before} after={after}")

    # the honest receipt's leaf was NOT invalidated (only the fraudulent one).
    # (the batch's single invalidation == exactly the fraud leaf value asserted above.)
    assert hash_leaf(batched_receipt_to_leaf(honest)) != \
        hash_leaf(batched_receipt_to_leaf(fraud))


@requires_hardhat
def test_invalid_signature_multileaf_proof_onchain_e2e(hardhat_node, deployed):
    """Same operator path, but a MULTI-LEAF batch so the on-chain challenge verifies a
    NON-TRIVIAL OpenZeppelin sorted-pair merkle proof (the single-leaf case has an empty
    proof). This is the on-chain validation of the proof CONVENTION — the class of mismatch
    a leaf/proof bug (like sp1165's) would surface here but not in a single-leaf or unit test."""
    from web3 import Web3

    from prsm.economy.web3.batch_settlement_contract_client import (
        Web3SettlementContractClient,
    )
    from prsm.node.identity import generate_node_identity
    from prsm.settlement.accumulator import AccumulatorConfig, ReceiptAccumulator
    from prsm.settlement.challenge_broadcaster import gated_broadcast_prepared
    from prsm.settlement.challenge_preparer import prepare_challenge_from_finding
    from prsm.settlement.client import BatchSettlementClient
    from prsm.settlement.invalid_signature_submitter import ChallengeSubmitter
    from prsm.settlement.merkle import (
        batched_receipt_to_leaf,
        hash_leaf,
        verify_merkle_proof,
    )
    from prsm.settlement.settlement_audit_report import AuditFindingRecord

    requester = Web3.to_checksum_address(deployed["requester"])
    provider = Web3.to_checksum_address(deployed["node0"])
    idn = generate_node_identity(display_name="sp1167-multileaf")

    # 3 receipts: indexes 0,1 HONEST + index 2 FRAUDULENT -> a 3-leaf tree, so the fraud
    # leaf's proof is non-trivial (exercises odd-leaf promotion + sorted-pair on-chain).
    receipts = [
        _batched_receipt(idn, requester=requester, provider=provider, job="J", shard=0,
                         valid=True, out=("a1" * 32), escrow="m0"),
        _batched_receipt(idn, requester=requester, provider=provider, job="J", shard=1,
                         valid=True, out=("a2" * 32), escrow="m1"),
        _batched_receipt(idn, requester=requester, provider=provider, job="J", shard=2,
                         valid=False, out=("a3" * 32), escrow="m2"),
    ]
    fraud_index = 2
    fraud_leaf_value = receipts[fraud_index].value_ftns

    contract_client = Web3SettlementContractClient(
        rpc_url=hardhat_node, contract_address=deployed["registry"],
        private_key=deployed["node0Key"])
    client = BatchSettlementClient(
        accumulator=ReceiptAccumulator(
            # value threshold astronomically high (wei) so ONLY count_threshold=3 triggers
            # -> all three commit together in insertion order.
            AccumulatorConfig(count_threshold=3, time_threshold_seconds=3600,
                              value_threshold_ftns=10 ** 30)),
        contract_client=contract_client, provider_address=provider)

    async def _commit():
        for br in receipts:
            await client.accumulate(br)
        return await client.commit_ready_batches()

    committed = asyncio.run(_commit())
    assert len(committed) == 1 and committed[0].receipt_count == 3, (
        f"expected 1 batch of 3 receipts, got "
        f"{len(committed)} / {committed[0].receipt_count if committed else '-'}")
    batch_id = committed[0].batch_id
    bid_hex = batch_id.hex()

    record = AuditFindingRecord(
        reason="invalid_signature", batch_id_hex=bid_hex, target_index=fraud_index,
        on_chain_reason=1, detail={"receipt_index": fraud_index})
    prepared = prepare_challenge_from_finding(
        record=record,
        receipt_lookup=lambda b: list(receipts)
        if str(b).lower().removeprefix("0x") == bid_hex else None)

    # the proof is NON-trivial (a 3-leaf tree) and verifies against the on-chain root
    assert len(prepared.challenge.merkle_proof) >= 1, "expected a non-trivial merkle proof"
    onchain_root = bytes(contract_client.contract.functions.batches(batch_id).call()[2])
    assert verify_merkle_proof(
        prepared.challenge.merkle_proof, onchain_root, hash_leaf(prepared.challenge.leaf))

    submitter = ChallengeSubmitter(
        rpc_url=hardhat_node, registry_address=deployed["registry"],
        private_key=deployed["challengerKey"])
    before = int(contract_client.contract.functions.batches(batch_id).call()[5])
    outcome = gated_broadcast_prepared(
        prepared, submitter=submitter, broadcast_enabled=True)

    assert outcome.dry_run_ok is True, (
        f"multi-leaf dry-run should pass on-chain: {outcome.dry_run_reason}")
    assert outcome.broadcast is True, f"challenge should broadcast: {outcome.error_message}"
    after = int(contract_client.contract.functions.batches(batch_id).call()[5])
    assert after - before == fraud_leaf_value, (
        f"fraud receipt must be invalidated on-chain via the NON-TRIVIAL proof; "
        f"before={before} after={after}")
