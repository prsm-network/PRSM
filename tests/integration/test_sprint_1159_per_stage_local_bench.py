"""Sprint 1159 — LOCAL HARDHAT BENCH for the on-chain per-stage payee arc
(BRICK 4 + end-to-end validation).

Boots a local Hardhat node, deploys the FULL settlement suite (MockERC20/FTNS +
EscrowPool + MockSignatureVerifier + BatchSettlementRegistry, in the order the
phase7 e2e mandates — Registry BEFORE Pool because EscrowPool bakes
``settlementRegistry`` immutable — cross-wired with setEscrowPool /
setSignatureVerifier), and drives the WHOLE arc end-to-end against the REAL EVM
with the REAL deployed contracts (NOT mocked web3):

  (a) deploy the suite to local hardhat;
  (b) map hardhat accounts to the requester + 3 worker NODES — each node has an
      Ed25519 identity (for the challenge-defensible leaf signature) AND a
      DISTINCT hardhat eth key (for committing its own share-batch);
  (c) the requester deposits the TOTAL into EscrowPool;
  (d) build a multi-stage InferenceReceipt + each node signs its per-stage leaf
      (reusing brick-1's signing approach), then split into per-node share-batches
      via brick 1 (``split_receipt_to_per_node_batched_receipts``);
  (e) drive BRICK 4: each node commits + finalizes its OWN share-batch (the
      ``per_stage_commit`` orchestration); ``evm_increaseTime`` past the 1-hour
      challenge window so finalize is reachable in-test;
  (f) ASSERT the on-chain money-safety properties:
        * each node's FTNS wallet balance increased by EXACTLY its brick-1
          conserving share (no node over/under-paid);
        * the requester's EscrowPool balance drained by EXACTLY the total
          (== sum of shares; no over-draw);
        * every batch reads FINALIZED (status == 2) exactly once (no double-settle);
        * the registry's per-batch ``provider`` == the committing node's address.

The REAL contracts move REAL (test) FTNS through the REAL EscrowPool — this is
the on-chain payout validation, not a Python-side mock.

Skipped automatically if ``npx hardhat`` is not on PATH / node_modules absent.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import socket
import subprocess
import time
from decimal import Decimal
from pathlib import Path

import pytest

# conftest patches subprocess/time for the session; capture the real Popen now.
from subprocess import Popen as _REAL_POPEN  # noqa: E402

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
    """Local hardhat node for the module on port 8545 (== hardhat's built-in
    ``localhost`` network URL, so the inline deploy can use ``--network
    localhost`` exactly like the phase7 e2e). Module-scoped; pytest runs modules
    serially by default — do NOT xdist this against the phase7 e2e (both bind
    :8545)."""
    log_path = Path("/tmp/prsm_sp1159_per_stage_hardhat.log")
    log_fh = open(log_path, "wb")
    proc = _REAL_POPEN(
        "exec npx hardhat node --port 8545",
        cwd=str(CONTRACTS_DIR),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        shell=True,
        start_new_session=True,
    )
    try:
        deadline = time.time() + 120.0
        port_open = False
        while time.time() < deadline:
            if proc.poll() is not None:
                log_fh.close()
                tail = log_path.read_text(errors="replace")[-2000:]
                raise RuntimeError(
                    f"hardhat node exited early (rc={proc.returncode})\n{tail}"
                )
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
            tail = log_path.read_text(errors="replace")[-2000:]
            raise TimeoutError(
                f"hardhat node did not open port 8545 within 120s\n{tail}"
            )
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
    """Compile + deploy the full settlement suite via an inline JS script.

    Order MIRRORS the phase7 e2e: Token -> Registry (1-hour window, the min) ->
    Pool (registry baked immutable) -> setEscrowPool -> Verifier ->
    setSignatureVerifier. StakeBond is NOT needed here (no slashing path), and
    tier_slash_rate_bps==0 means commitBatch never touches the bond.

    Accounts: hardhat's well-known pre-funded keys. We use:
      deployer=acct0, requester=acct1, node0=acct2, node1=acct3, node2=acct4.
    The requester is minted + deposits the TOTAL into escrow; each node commits
    with its OWN key (msg.sender => b.provider == that node).
    """
    deploy_js = """
const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const signers = await hre.ethers.getSigners();
  const [deployer, requester, node0, node1, node2] = signers;

  const Token = await hre.ethers.getContractFactory("MockERC20");
  const token = await Token.deploy();
  await token.waitForDeployment();

  // Registry BEFORE Pool (EscrowPool bakes settlementRegistry immutable).
  // 1-hour challenge window == MIN_CHALLENGE_WINDOW_SECONDS.
  const Registry = await hre.ethers.getContractFactory(
    "BatchSettlementRegistry"
  );
  const registry = await Registry.deploy(deployer.address, 3600);
  await registry.waitForDeployment();

  const Pool = await hre.ethers.getContractFactory("EscrowPool");
  const pool = await Pool.deploy(
    deployer.address,
    await token.getAddress(),
    await registry.getAddress(),
  );
  await pool.waitForDeployment();
  await registry.connect(deployer).setEscrowPool(await pool.getAddress());

  // Mock verifier defaults to returning false; with NO challenges fired in this
  // bench it is never exercised on the money path — present only because the
  // registry requires a configured verifier for the commit/finalize wiring.
  const Verifier = await hre.ethers.getContractFactory(
    "MockSignatureVerifier"
  );
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();
  await registry.connect(deployer).setSignatureVerifier(
    await verifier.getAddress()
  );

  const out = {
    token: await token.getAddress(),
    pool: await pool.getAddress(),
    registry: await registry.getAddress(),
    verifier: await verifier.getAddress(),
    deployer: deployer.address,
    requester: requester.address,
    nodes: [node0.address, node1.address, node2.address],
    // Hardhat default account private keys (well-known, testnet-only).
    requesterKey:
      "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    nodeKeys: [
      "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
      "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
      "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
    ],
    deployerKey:
      "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
  };
  fs.writeFileSync(
    "/tmp/prsm_sp1159_per_stage_deploy.json", JSON.stringify(out)
  );
}
main().catch((e) => { console.error(e); process.exit(1); });
"""
    script = CONTRACTS_DIR / "scripts" / "_sp1159_per_stage_deploy.js"
    script.write_text(deploy_js)

    def _run(argv):
        p = _REAL_POPEN(argv, cwd=str(CONTRACTS_DIR))
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"command failed (rc={rc}): {' '.join(argv)}")

    try:
        _run(["npx", "hardhat", "compile"])
        _run([
            "npx", "hardhat", "run", "scripts/_sp1159_per_stage_deploy.js",
            "--network", "localhost",
        ])
        with open("/tmp/prsm_sp1159_per_stage_deploy.json") as f:
            return json.load(f)
    finally:
        if script.exists():
            script.unlink()


def _make_multistage_receipt(node_ids):
    """A 3-stage InferenceReceipt whose topology names the 3 distinct nodes."""
    from prsm.compute.inference.models import InferenceReceipt, ContentTier
    from prsm.compute.inference.topology_rotation import TopologyAssignment
    from prsm.compute.tee.models import PrivacyLevel, TEEType

    output_hash = hashlib.sha256(b"sp1159 multi-stage on-chain output").digest()
    topo = TopologyAssignment(
        positions={(i, 0): nid for i, nid in enumerate(node_ids)},
        stage_count=len(node_ids),
        slots_per_stage=1,
    )
    return InferenceReceipt(
        job_id="job-sp1159",
        request_id="req-sp1159-onchain-bench",
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att",
        output_hash=output_hash,
        duration_seconds=1.0,
        cost_ftns=Decimal("1.0"),
        settler_signature=b"parallax-signature-bytes",
        settler_node_id=node_ids[0],
        topology_assignment=topo,
    )


def _sig_material_for(identity, *, job_id, stage_index, output_hash_hex, executed_at_unix):
    from prsm.compute.shard_receipt import build_receipt_signing_payload
    from prsm.settlement.per_stage_settlement_split import NodeSignatureMaterial

    payload = build_receipt_signing_payload(
        job_id=job_id, shard_index=stage_index,
        output_hash=output_hash_hex, executed_at_unix=executed_at_unix,
    )
    return NodeSignatureMaterial(
        pubkey_b64=identity.public_key_b64,
        signature=identity.sign(payload),
        stage_index=stage_index,
        output_hash=output_hash_hex,
        executed_at_unix=executed_at_unix,
    )


@requires_hardhat
def test_sp1159_per_stage_onchain_payout_e2e(hardhat_node, deployed):
    """Full per-stage on-chain payout: split -> per-node commit -> finalize ->
    assert each node paid EXACTLY its conserving share + escrow drained by the
    total + every batch FINALIZED once (no over-draw / no double-settle)."""
    from web3 import Web3

    from prsm.compute.shard_receipt import per_stage_leaf_job_id
    from prsm.node.identity import generate_node_identity
    from prsm.settlement.accumulator import (
        AccumulatorConfig,
        ReceiptAccumulator,
    )
    from prsm.settlement.client import BatchSettlementClient
    from prsm.economy.web3.batch_settlement_contract_client import (
        BATCH_SETTLEMENT_REGISTRY_ABI,
        Web3SettlementContractClient,
    )
    from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
    from prsm.settlement.per_stage_commit import (
        commit_per_node_share_batches,
        finalize_per_node_share_batches,
    )
    from prsm.settlement.per_stage_settlement_split import (
        split_receipt_to_per_node_batched_receipts,
    )

    w3 = Web3(Web3.HTTPProvider(hardhat_node))

    # ── (b) account mapping: 3 worker nodes, each Ed25519 identity + eth key ──
    identities = [generate_node_identity(display_name=f"sp1159-stage{i}")
                  for i in range(3)]
    node_ids = [idn.node_id for idn in identities]
    node_eth = {node_ids[i]: deployed["nodes"][i] for i in range(3)}
    node_keys = {node_ids[i]: deployed["nodeKeys"][i] for i in range(3)}

    requester = deployed["requester"]
    total_wei = 30 * _ONE_FTNS + 1  # non-divisible by 3 -> exercises remainder

    # ── (c) requester mints + deposits the TOTAL into EscrowPool ──────────────
    # Mint FTNS to the requester (MockERC20 has a public mint), then deposit.
    erc20_mint_abi = [{
        "inputs": [{"name": "to", "type": "address"},
                   {"name": "amount", "type": "uint256"}],
        "name": "mint", "outputs": [],
        "stateMutability": "nonpayable", "type": "function",
    }]
    token_mint = w3.eth.contract(
        address=Web3.to_checksum_address(deployed["token"]), abi=erc20_mint_abi)
    deployer_acct = w3.eth.account.from_key(deployed["deployerKey"])
    mint_tx = token_mint.functions.mint(
        Web3.to_checksum_address(requester), total_wei,
    ).build_transaction({
        "from": deployer_acct.address,
        "nonce": w3.eth.get_transaction_count(deployer_acct.address, "pending"),
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id,
    })
    signed = w3.eth.account.sign_transaction(mint_tx, deployer_acct.key)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    rcpt = w3.eth.wait_for_transaction_receipt(
        w3.eth.send_raw_transaction(raw), timeout=60)
    assert rcpt.status == 1, "mint to requester reverted"

    escrow = EscrowPoolClient(
        rpc_url=hardhat_node,
        escrow_pool_address=deployed["pool"],
        ftns_token_address=deployed["token"],
        private_key=deployed["requesterKey"],
    )
    asyncio.run(escrow.deposit(total_wei))
    escrow_before = asyncio.run(escrow.balance_of(requester))
    assert escrow_before == total_wei, (
        f"requester escrow should be the deposited total {total_wei}, "
        f"got {escrow_before}"
    )

    # ── (d) build the multi-stage receipt + per-node leaf sigs, split (brick 1) ─
    ir = _make_multistage_receipt(node_ids)
    output_hex = ir.output_hash.hex()
    executed = int(time.time())
    job_id = per_stage_leaf_job_id(ir.request_id)
    node_sigs = {
        node_ids[i]: _sig_material_for(
            identities[i], job_id=job_id, stage_index=i,
            output_hash_hex=output_hex, executed_at_unix=executed,
        )
        for i in range(3)
    }
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total_wei,
        node_signatures=node_sigs,
        requester_address=Web3.to_checksum_address(requester),
    )
    assert shares is not None and len(shares) == 3, "brick-1 split must produce 3 shares"
    # Conservation (offline): the per-node shares sum to the total EXACTLY.
    assert sum(s.share_wei for s in shares) == total_wei
    expected_share_by_node = {s.node_id: s.share_wei for s in shares}

    # ── build one BatchSettlementClient per node (its OWN key + bound address) ──
    # A small value threshold so each node's single share-receipt fires a commit
    # immediately (the conserving share >> threshold). This is the deliberate,
    # caller-driven commit cadence — no background loop.
    cfg = AccumulatorConfig(
        count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=1,
    )
    clients = {}
    for nid in node_ids:
        committer = Web3.to_checksum_address(node_eth[nid])
        contract_client = Web3SettlementContractClient(
            rpc_url=hardhat_node,
            contract_address=deployed["registry"],
            private_key=node_keys[nid],
        )
        clients[nid] = BatchSettlementClient(
            accumulator=ReceiptAccumulator(cfg),
            contract_client=contract_client,
            provider_address=committer,
        )

    def client_for_node(nid):
        return clients[nid]

    def committer_for_node(nid):
        return Web3.to_checksum_address(node_eth[nid])

    # FTNS wallet balances BEFORE settlement (each node's wallet should rise by
    # EXACTLY its share after finalize).
    ftns_before = {
        nid: asyncio.run(escrow.ftns_balance_of(node_eth[nid])) for nid in node_ids
    }

    # ── (e) BRICK 4: each node commits its OWN share-batch ────────────────────
    committed = asyncio.run(commit_per_node_share_batches(
        shares=shares,
        client_for_node=client_for_node,
        committer_address_for_node=committer_for_node,
    ))
    assert set(committed.keys()) == set(node_ids), (
        f"every node must commit its share-batch; got {set(committed.keys())}"
    )

    # On-chain provider-of-record per batch == the committing node (msg.sender).
    registry_read = w3.eth.contract(
        address=Web3.to_checksum_address(deployed["registry"]),
        abi=BATCH_SETTLEMENT_REGISTRY_ABI,
    )
    batch_ids = {}
    for nid in node_ids:
        cb = committed[nid]
        batch_ids[nid] = cb.batch_id
        b = registry_read.functions.batches(cb.batch_id).call()
        assert b[0].lower() == node_eth[nid].lower(), (
            f"on-chain provider for {nid}'s batch must be its committer "
            f"{node_eth[nid]}, got {b[0]}"
        )
        assert b[1].lower() == requester.lower(), "batch requester must be the funder"
        assert int(b[4]) == expected_share_by_node[nid], (
            "committed totalValueFTNS must equal the node's conserving share"
        )
        assert int(b[7]) == 1, "batch must be PENDING before finalize"

    # ── time-travel past the 1-hour challenge window so finalize is reachable ──
    w3.provider.make_request("evm_increaseTime", [3601])
    w3.provider.make_request("evm_mine", [])

    # ── BRICK 4: each node finalizes its OWN share-batch (escrow -> node) ──────
    finalized = asyncio.run(finalize_per_node_share_batches(
        committed=committed,
        client_for_node=client_for_node,
    ))
    assert set(finalized.keys()) == set(node_ids), (
        f"every node must finalize its batch; got {set(finalized.keys())}"
    )

    # ── (f) MONEY-SAFETY ASSERTIONS on the REAL chain ─────────────────────────
    # 1. Each node's FTNS wallet rose by EXACTLY its brick-1 conserving share.
    total_paid = 0
    for nid in node_ids:
        after = asyncio.run(escrow.ftns_balance_of(node_eth[nid]))
        delta = after - ftns_before[nid]
        assert delta == expected_share_by_node[nid], (
            f"node {nid} paid {delta} wei on-chain but its conserving share is "
            f"{expected_share_by_node[nid]} — over/under-paid"
        )
        total_paid += delta

    # 2. Sum of on-chain payouts == the total (no FTNS created/lost).
    assert total_paid == total_wei, (
        f"sum of on-chain payouts {total_paid} != total {total_wei}"
    )

    # 3. Requester escrow drained by EXACTLY the total (no over-draw).
    escrow_after = asyncio.run(escrow.balance_of(requester))
    assert escrow_after == escrow_before - total_wei == 0, (
        f"requester escrow must drain by exactly the total: before={escrow_before} "
        f"after={escrow_after} total={total_wei}"
    )

    # 4. Every batch reads FINALIZED (status==2) exactly once; a second finalize
    #    pass is a no-op (idempotent — no double-settle).
    for nid in node_ids:
        b = registry_read.functions.batches(batch_ids[nid]).call()
        assert int(b[7]) == 2, f"{nid}'s batch must be FINALIZED (status 2), got {b[7]}"

    second_pass = asyncio.run(finalize_per_node_share_batches(
        committed=committed, client_for_node=client_for_node,
    ))
    assert second_pass == {}, "a second finalize pass must be a no-op (no double-settle)"

    # Balances unchanged after the no-op second pass (double-settle would move FTNS).
    for nid in node_ids:
        after2 = asyncio.run(escrow.ftns_balance_of(node_eth[nid]))
        assert after2 - ftns_before[nid] == expected_share_by_node[nid], (
            f"node {nid} balance changed on the idempotent second finalize pass — "
            f"double-settle"
        )
    assert asyncio.run(escrow.balance_of(requester)) == 0

    # 5. CONTRACT-LEVEL double-settle guard. The Python client's idempotency (above) is
    #    one layer; assert the on-chain guard too — a RAW finalizeBatch on an
    #    already-FINALIZED batch, BYPASSING the client, must REVERT (BatchNotPending).
    #    Without this, a regression in the contract guard would not be caught by the
    #    Python-idempotency assertion alone.
    _finalize_only = w3.eth.contract(
        address=Web3.to_checksum_address(deployed["registry"]),
        abi=[{
            "type": "function", "name": "finalizeBatch",
            "stateMutability": "nonpayable",
            "inputs": [{"name": "batchId", "type": "bytes32"}], "outputs": [],
        }],
    )
    _probe_nid = node_ids[0]
    _probe_acct = w3.eth.account.from_key(node_keys[_probe_nid])
    _reverted = False
    try:
        _raw_tx = _finalize_only.functions.finalizeBatch(
            batch_ids[_probe_nid]
        ).build_transaction({
            "from": _probe_acct.address,
            "nonce": w3.eth.get_transaction_count(_probe_acct.address),
            "gas": 500_000,  # explicit gas → mine-and-revert (status 0), not estimation-raise
            "gasPrice": w3.eth.gas_price,
            "chainId": w3.eth.chain_id,
        })
        _signed = w3.eth.account.sign_transaction(_raw_tx, _probe_acct.key)
        _raw = getattr(_signed, "raw_transaction", None) or getattr(
            _signed, "rawTransaction", None)
        _rcpt = w3.eth.wait_for_transaction_receipt(
            w3.eth.send_raw_transaction(_raw), timeout=60)
        _reverted = int(_rcpt.status) == 0
    except Exception:
        _reverted = True  # estimation/broadcast reverted (BatchNotPending) — guard holds
    assert _reverted, (
        "a RAW on-chain finalizeBatch on an already-FINALIZED batch must REVERT "
        "(contract-level double-settle guard — not just the Python client's idempotency)"
    )
    # The reverted re-finalize moved nothing: batch still FINALIZED, payout still exact.
    assert int(registry_read.functions.batches(batch_ids[_probe_nid]).call()[7]) == 2
    assert asyncio.run(
        escrow.ftns_balance_of(node_eth[_probe_nid])
    ) - ftns_before[_probe_nid] == expected_share_by_node[_probe_nid]
