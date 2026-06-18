#!/usr/bin/env python3
"""Sprint 1160 — PER-STAGE Base-Sepolia settlement bench (operator-run counterpart
to the sp1159 local-hardhat bench).

THE ARC (project_onchain_per_stage_payee_arc). On a multi-stage cross-host
inference we pay EACH topology node its share ON-CHAIN. Bricks 1-4 are COMPLETE
and validated on a REAL EVM via the sp1159 LOCAL-hardhat bench
(tests/integration/test_sprint_1159_per_stage_local_bench.py). THIS script lets
the OPERATOR run the SAME proven flow against Base Sepolia (real testnet) with
their OWN funded keys — it reinvents NOTHING:

  * brick 1 (per_stage_settlement_split.split_receipt_to_per_node_batched_receipts)
    splits a multi-stage InferenceReceipt into conserving, challenge-defensible
    per-node share-batches (the leaf sig preimage is
    shard_receipt.build_receipt_signing_payload over per_stage_leaf_job_id(request_id)).
  * brick 4 (per_stage_commit.commit_per_node_share_batches /
    finalize_per_node_share_batches) — each NODE commits + finalizes its OWN
    share-batch, keyed by its OWN eth key (msg.sender => on-chain provider == node).
  * the settlement clients (EscrowPoolClient deposit, BatchSettlementClient
    commit/finalize, Web3SettlementContractClient) are RPC-agnostic — the SAME
    flow runs on Sepolia by pointing at BASE_SEPOLIA_RPC_URL + the Sepolia addresses
    resolved from prsm.config.networks.

This MIRRORS scripts/settlement_sepolia_e2e.py (sp1042): two-phase (deposit+commit
now, finalize after the challenge window), PRSM_NETWORK=testnet default, mainnet
HARD-GUARD (refuses chainId 8453 without --i-understand-mainnet), durable state
between phases, read-only verify. The harness PREPARES + VERIFIES; the OPERATOR
runs anything that broadcasts (keys in THEIR env, NEVER embedded/shared/logged).

  PHASE 1 (now) — requester deposits the TOTAL; each node commits its share-batch:
    cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet \
      PRSM_REQUESTER_KEY=0x<A> PRSM_NODE_KEYS=0x<n0>,0x<n1>,0x<n2> \
      python scripts/per_stage_sepolia_bench.py --phase deposit-commit --amount-ftns 1

  PHASE 2 (after the ~24h challenge window) — each node finalizes its share-batch:
    cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet \
      PRSM_NODE_KEYS=0x<n0>,0x<n1>,0x<n2> \
      python scripts/per_stage_sepolia_bench.py --phase finalize

  VERIFY (anytime; read-only, NO keys) — resolve addresses + read batch state:
    cd ~/Documents/GitHub/PRSM && PRSM_NETWORK=testnet \
      python scripts/per_stage_sepolia_bench.py --phase verify

Keys: PRSM_REQUESTER_KEY (the paying requester A — deposits the TOTAL into its OWN
escrow) + PRSM_NODE_KEYS (comma-separated node eth keys — each node commits +
finalizes ITS share-batch). EVERY key comes from the operator env. NO key is
embedded, defaulted to a real key, or logged — the script echoes the signer
ADDRESS, never the key (the sp1046 lesson). The requester key is NOT needed for
phase 2 (the committed batches already name their requester + node providers).

Exit codes: 0 success/committed; 1 runtime failure; 2 env/validation error.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Base chain ids — re-exported from e2e_proof's single source of truth so the
# guard is identical to settlement_sepolia_e2e.py.
from prsm.settlement.e2e_proof import (  # noqa: E402
    BASE_MAINNET_CHAIN_ID,
    BASE_SEPOLIA_CHAIN_ID,
    assert_chain_allowed,
)

_ONE_FTNS = 10 ** 18


# ──────────────────────────────────────────────────────────────────────────────
# KEY GATING — every key from the operator env; NONE embedded/defaulted/logged.
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_key(raw: str) -> str:
    raw = raw.strip()
    return raw if raw.startswith("0x") else "0x" + raw


def read_requester_key(env: Optional[Dict[str, str]] = None) -> str:
    """The paying requester A's private key — REQUIRED for phase 1 (deposit). Read
    ONLY from the operator env (PRSM_REQUESTER_KEY, or the REQUESTER_PRIVATE_KEY
    alias that settlement_sepolia_e2e.py uses). NEVER embedded / defaulted to a
    real key. Raises SystemExit(2) with a clear message when unset."""
    env = os.environ if env is None else env
    for name in ("PRSM_REQUESTER_KEY", "REQUESTER_PRIVATE_KEY"):
        v = (env.get(name, "") or "").strip()
        if v:
            return _normalize_key(v)
    print("ERROR: set PRSM_REQUESTER_KEY (or REQUESTER_PRIVATE_KEY) to the paying "
          "requester A's funded key (it deposits the TOTAL into its own escrow). "
          "No key is embedded or defaulted — it must come from your env.",
          file=sys.stderr)
    raise SystemExit(2)


def read_node_keys(env: Optional[Dict[str, str]] = None) -> List[str]:
    """The per-node eth private keys — REQUIRED for both phases (each node commits +
    finalizes its OWN share-batch). Read ONLY from the operator env: PRSM_NODE_KEYS
    (comma-separated) OR the indexed PRSM_NODE_KEY_0 / PRSM_NODE_KEY_1 / ... aliases.
    NEVER embedded / defaulted to a real key. Raises SystemExit(2) when none set."""
    env = os.environ if env is None else env
    csv = (env.get("PRSM_NODE_KEYS", "") or "").strip()
    if csv:
        keys = [_normalize_key(k) for k in csv.split(",") if k.strip()]
        if keys:
            return keys
    indexed: List[str] = []
    i = 0
    while True:
        v = (env.get(f"PRSM_NODE_KEY_{i}", "") or "").strip()
        if not v:
            break
        indexed.append(_normalize_key(v))
        i += 1
    if indexed:
        return indexed
    print("ERROR: set PRSM_NODE_KEYS (comma-separated) or PRSM_NODE_KEY_0, "
          "PRSM_NODE_KEY_1, ... to the per-node funded eth keys (each node commits "
          "+ finalizes its own share-batch). No key is embedded or defaulted — they "
          "must come from your env. The bench needs >= 2 node keys.",
          file=sys.stderr)
    raise SystemExit(2)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE-1 ASSEMBLY — build the multi-stage receipt + per-node leaf sigs + brick-1
# split. PURE (no chain, no broadcast); offline-testable like e2e_proof's helpers.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageBenchAssembly:
    """The offline-assembled artefacts phase 1 hands to the chain step: the per-node
    brick-1 share-batches, the eth address each node will commit with, and the
    request_id / total for the report."""

    request_id: str
    total_wei: int
    shares: list  # List[StageSettlementShare]
    node_ids: List[str]
    node_eth: Dict[str, str]          # node_id -> committer eth address
    expected_share_by_node: Dict[str, int]


def _make_multistage_receipt(node_ids: List[str], *, request_id: str, output_hash: bytes):
    """An N-stage InferenceReceipt whose topology names the N distinct nodes — the
    SAME receipt shape the sp1159 local bench uses (one node per stage)."""
    from prsm.compute.inference.models import InferenceReceipt, ContentTier
    from prsm.compute.inference.topology_rotation import TopologyAssignment
    from prsm.compute.tee.models import PrivacyLevel, TEEType

    topo = TopologyAssignment(
        positions={(i, 0): nid for i, nid in enumerate(node_ids)},
        stage_count=len(node_ids),
        slots_per_stage=1,
    )
    return InferenceReceipt(
        job_id=f"job-{request_id}",
        request_id=request_id,
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


def assemble_phase1(
    *,
    node_keys: List[str],
    requester_address: str,
    total_wei: int,
    request_id: Optional[str] = None,
    output_hash: Optional[bytes] = None,
    executed_at_unix: Optional[int] = None,
) -> StageBenchAssembly:
    """Build the multi-stage receipt, each node's per-stage leaf signature, and the
    brick-1 per-node share-batch split — EXACTLY as the sp1159 local bench does, but
    for the operator-supplied node keys. PURE: derives one fresh Ed25519 identity per
    node (the challenge-defensible leaf signer), maps each to its eth committer
    address, signs each node's per-stage leaf over build_receipt_signing_payload(
    per_stage_leaf_job_id(request_id), stage_index, output_hash, executed_at_unix),
    and calls brick 1. NEVER signs an eth tx / touches the chain.

    Raises ValueError if fewer than 2 node keys (per-stage needs >= 2 distinct
    nodes — brick 1 returns None below 2, which is the single-payee fallback, not a
    valid per-stage bench)."""
    from eth_account import Account
    from web3 import Web3

    from prsm.compute.shard_receipt import (
        build_receipt_signing_payload,
        per_stage_leaf_job_id,
    )
    from prsm.node.identity import generate_node_identity
    from prsm.settlement.per_stage_settlement_split import (
        NodeSignatureMaterial,
        split_receipt_to_per_node_batched_receipts,
    )

    if len(node_keys) < 2:
        raise ValueError(
            f"per-stage bench needs >= 2 node keys (got {len(node_keys)}); below 2 "
            f"distinct nodes the split falls back to the single-payee path."
        )

    n = len(node_keys)
    identities = [generate_node_identity(display_name=f"sp1160-sepolia-stage{i}")
                  for i in range(n)]
    node_ids = [idn.node_id for idn in identities]
    node_eth = {
        node_ids[i]: Web3.to_checksum_address(Account.from_key(node_keys[i]).address)
        for i in range(n)
    }

    request_id = request_id or f"sp1160-sepolia-{int(time.time())}"
    if output_hash is None:
        output_hash = hashlib.sha256(
            f"sp1160 per-stage sepolia bench {request_id}".encode()).digest()
    output_hex = output_hash.hex()
    executed = int(executed_at_unix if executed_at_unix is not None else time.time())
    job_id = per_stage_leaf_job_id(request_id)

    node_sigs = {}
    for i in range(n):
        payload = build_receipt_signing_payload(
            job_id=job_id, shard_index=i,
            output_hash=output_hex, executed_at_unix=executed,
        )
        node_sigs[node_ids[i]] = NodeSignatureMaterial(
            pubkey_b64=identities[i].public_key_b64,
            signature=identities[i].sign(payload),
            stage_index=i,
            output_hash=output_hex,
            executed_at_unix=executed,
        )

    ir = _make_multistage_receipt(node_ids, request_id=request_id, output_hash=output_hash)
    shares = split_receipt_to_per_node_batched_receipts(
        receipt=ir,
        total_value_wei=total_wei,
        node_signatures=node_sigs,
        requester_address=Web3.to_checksum_address(requester_address),
    )
    if shares is None:
        raise RuntimeError(
            "brick-1 split returned None (topology/signature gating) — the bench "
            "receipt should always split; this indicates an assembly bug."
        )
    # Money-safety (offline): the per-node shares conserve the total EXACTLY.
    assert sum(s.share_wei for s in shares) == total_wei, "split broke conservation"

    return StageBenchAssembly(
        request_id=request_id,
        total_wei=total_wei,
        shares=shares,
        node_ids=node_ids,
        node_eth=node_eth,
        expected_share_by_node={s.node_id: s.share_wei for s in shares},
    )


# ──────────────────────────────────────────────────────────────────────────────
# READ-ONLY VERIFY LOGIC — pure computation of the expected per-node payouts.
# ──────────────────────────────────────────────────────────────────────────────


def compute_expected_payouts(assembly: StageBenchAssembly) -> Dict[str, int]:
    """The expected on-chain payout per committer eth address == that node's
    conserving brick-1 share. PURE — the read-only verify compares these against the
    live wallet deltas. Asserts conservation (sum == total)."""
    by_eth: Dict[str, int] = {}
    for nid in assembly.node_ids:
        by_eth[assembly.node_eth[nid]] = assembly.expected_share_by_node[nid]
    assert sum(by_eth.values()) == assembly.total_wei, (
        "expected payouts must conserve the total"
    )
    return by_eth


def _amount_to_wei(amount_ftns: float) -> int:
    # Decimal so e.g. 0.1 FTNS is exact wei, not a float-rounding artefact.
    return int((Decimal(str(amount_ftns)) * _ONE_FTNS).to_integral_value())


# ──────────────────────────────────────────────────────────────────────────────
# STATE — durable bridge between phase 1 and phase 2 (the committed batch_ids +
# their committer addresses). Mirrors the settlement_sepolia_e2e state-file idea.
# ──────────────────────────────────────────────────────────────────────────────


def _default_state_file() -> str:
    from pathlib import Path
    return str(Path.home() / ".prsm" / "per_stage_sepolia_bench_state.json")


def _save_state(path: str, state: dict) -> None:
    import json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def _load_state(path: str) -> Optional[dict]:
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ──────────────────────────────────────────────────────────────────────────────
# CHAIN STEPS — each broadcasts ONLY with the operator-supplied keys.
# ──────────────────────────────────────────────────────────────────────────────


def _build_node_clients(*, rpc_url: str, registry: str, assembly: StageBenchAssembly,
                        node_keys: List[str]):
    """One BatchSettlementClient per node, bound to + keyed by that node's eth
    address — EXACTLY the sp1159 local-bench wiring (a small value threshold so each
    node's single share-receipt fires a commit immediately; no background loop)."""
    from web3 import Web3
    from prsm.settlement.accumulator import AccumulatorConfig, ReceiptAccumulator
    from prsm.settlement.client import BatchSettlementClient
    from prsm.economy.web3.batch_settlement_contract_client import (
        Web3SettlementContractClient,
    )

    key_by_eth = {
        Web3.to_checksum_address(__import__("eth_account").Account.from_key(k).address): k
        for k in node_keys
    }
    cfg = AccumulatorConfig(
        count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=1,
    )
    clients = {}
    for nid in assembly.node_ids:
        committer = assembly.node_eth[nid]
        contract_client = Web3SettlementContractClient(
            rpc_url=rpc_url, contract_address=registry,
            private_key=key_by_eth[committer],
        )
        clients[nid] = BatchSettlementClient(
            accumulator=ReceiptAccumulator(cfg),
            contract_client=contract_client,
            provider_address=committer,
        )
    return clients


async def _run_phase1(args, endpoints, chain_id) -> int:
    from eth_account import Account
    from web3 import Web3
    from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
    from prsm.settlement.e2e_proof import run_deposit_and_commit
    from prsm.settlement.per_stage_commit import commit_per_node_share_batches

    requester_key = read_requester_key()
    node_keys = read_node_keys()
    requester_addr = Web3.to_checksum_address(Account.from_key(requester_key).address)
    # Echo ADDRESSES, never keys (sp1046).
    node_addrs = [Web3.to_checksum_address(Account.from_key(k).address) for k in node_keys]
    print(f"requester(A)={requester_addr}")
    for i, a in enumerate(node_addrs):
        print(f"  node[{i}]={a}")
    if requester_addr.lower() in {a.lower() for a in node_addrs}:
        print("ERROR: the requester key must differ from every node key (a node "
              "paying itself from its own escrow is not a per-stage proof).",
              file=sys.stderr)
        return 2

    total_wei = _amount_to_wei(args.amount_ftns)
    if total_wei <= 0:
        print("ERROR: --amount-ftns must be positive", file=sys.stderr)
        return 2

    # Idempotency guard (mirror sp1042 review #6 / sp1048): if a prior phase-1 state
    # exists, refuse to re-deposit + re-commit (double-spend). Point at finalize.
    existing = _load_state(args.state_file)
    if existing and existing.get("committed"):
        print(f"ERROR: phase-1 state already present in {args.state_file} "
              f"({len(existing['committed'])} committed share-batch(es)). Phase 1 "
              f"already ran; re-depositing would double-spend. Run --phase finalize "
              f"after the challenge window, or delete the state file only if you know "
              f"the prior batches are dead.", file=sys.stderr)
        return 1

    assembly = assemble_phase1(
        node_keys=node_keys, requester_address=requester_addr, total_wei=total_wei)
    expected = compute_expected_payouts(assembly)
    print(f"assembled {len(assembly.shares)} per-node share-batches "
          f"(request_id={assembly.request_id}, total={total_wei} wei):")
    for nid in assembly.node_ids:
        print(f"  node {assembly.node_eth[nid]} share={assembly.expected_share_by_node[nid]} wei")

    # ── requester FUNDS ITS OWN claims into its escrow (idempotent, lag-tolerant) ──
    # sp1161 — the SHARED-ESCROW under-funding fix. EscrowPool.balanceOf is a SHARED
    # POOL per requester: a pre-existing balance may back OTHER pending batches (e.g.
    # a separate two-party batch), so a "skip if balance >= amount" deposit SILENTLY
    # UNDER-FUNDS — the per-stage finalize then reverts on insufficient escrow. By
    # DEFAULT we therefore deposit the per-stage total ADDITIVELY (fund our OWN
    # claims regardless of any pre-existing balance). The phase-1 commit-idempotency
    # guard above already refuses a re-deposit on a re-run, so a first-run additive
    # deposit is safe. --assume-dedicated-escrow opts back into the old skip-if-
    # sufficient behavior (ONLY safe for a KNOWN-dedicated escrow). NEVER silently
    # under-fund.
    escrow = EscrowPoolClient(
        rpc_url=endpoints.rpc_url, escrow_pool_address=endpoints.escrow_pool,
        ftns_token_address=endpoints.ftns_token, private_key=requester_key)
    pre_balance = int(await escrow.balance_of(requester_addr))
    if args.assume_dedicated_escrow and pre_balance >= total_wei:
        print(f"escrow already funded ({pre_balance} wei >= {total_wei}); skipping "
              f"deposit (--assume-dedicated-escrow: treating this escrow as dedicated "
              f"to this per-stage run)")
        escrow_balance = pre_balance
    else:
        if pre_balance > 0:
            print(f"WARNING: escrow already holds {pre_balance} wei which may back "
                  f"OTHER pending batches (the EscrowPool balance is a SHARED pool "
                  f"per requester); depositing the per-stage total ({total_wei} wei) "
                  f"ADDITIVELY so this bench funds its OWN claims — pass "
                  f"--assume-dedicated-escrow to skip if this escrow is dedicated.")
        target = pre_balance + total_wei  # additive: fund OUR claims on top.
        print(f"requester depositing {total_wei} wei additively into its escrow "
              f"(pre={pre_balance} -> target>={target}) …")
        deposit_tx = await escrow.deposit(total_wei)
        print(f"  deposit_tx={deposit_tx}")
        escrow_balance = pre_balance
        for _ in range(6):
            escrow_balance = int(await escrow.balance_of(requester_addr))
            if escrow_balance >= target:
                break
            await asyncio.sleep(args.poll_interval_s)
        if escrow_balance < target:
            print(f"ERROR: escrow underfunded after additive deposit ({escrow_balance} "
                  f"< {target}) even after polling past RPC lag; check the deposit "
                  f"tx on-chain before retrying", file=sys.stderr)
            return 1
    print(f"  escrow_balance={escrow_balance} wei")

    # ── BRICK 4: each node commits its OWN share-batch (msg.sender => provider) ─────
    clients = _build_node_clients(
        rpc_url=endpoints.rpc_url, registry=endpoints.settlement_registry,
        assembly=assembly, node_keys=node_keys)

    def client_for_node(nid):
        return clients[nid]

    def committer_for_node(nid):
        return assembly.node_eth[nid]

    committed = await commit_per_node_share_batches(
        shares=assembly.shares,
        client_for_node=client_for_node,
        committer_address_for_node=committer_for_node,
    )
    if set(committed.keys()) != set(assembly.node_ids):
        missing = set(assembly.node_ids) - set(committed.keys())
        print(f"ERROR: not every node committed its share-batch; missing "
              f"{len(missing)} (their receipts are retained/quarantined by their "
              f"client — never re-committed blindly). NOT a double-spend, but the "
              f"deposit is now partially un-finalizable; investigate before retry.",
              file=sys.stderr)
        # Persist what DID commit so phase 2 can still finalize those.
    state = {
        "network": endpoints.network_name,
        "chain_id": chain_id,
        "request_id": assembly.request_id,
        "total_wei": total_wei,
        "requester_address": requester_addr,
        "registry": endpoints.settlement_registry,
        "expected_payout_by_eth": expected,
        "committed": {
            nid: {
                "batch_id": committed[nid].batch_id.hex(),
                "committer_address": assembly.node_eth[nid],
                "share_wei": assembly.expected_share_by_node[nid],
            }
            for nid in committed
        },
    }
    _save_state(args.state_file, state)
    print(f"PHASE 1 done. {len(committed)} share-batch(es) committed; state saved to "
          f"{args.state_file}.")
    for nid in committed:
        print(f"  node {assembly.node_eth[nid]} batch_id={committed[nid].batch_id.hex()}")
    print(f"SAVE these batch_ids off-machine. After the challenge window run "
          f"--phase finalize (PRSM_NODE_KEYS only; the requester key is NOT needed).")
    return 0 if set(committed.keys()) == set(assembly.node_ids) else 1


async def _run_phase2(args, endpoints, chain_id) -> int:
    from eth_account import Account
    from web3 import Web3
    from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
    from prsm.economy.web3.batch_settlement_contract_client import (
        Web3SettlementContractClient,
    )
    from prsm.settlement.e2e_proof import run_finalize_by_batch_id

    state = _load_state(args.state_file)
    if not state or not state.get("committed"):
        print(f"ERROR: no phase-1 state with committed share-batches in "
              f"{args.state_file}; run --phase deposit-commit first (or restore the "
              f"state file).", file=sys.stderr)
        return 1

    node_keys = read_node_keys()
    addr_to_key = {
        Web3.to_checksum_address(Account.from_key(k).address): k for k in node_keys
    }
    for a in addr_to_key:
        print(f"  node signer={a}")  # ADDRESS, never the key

    escrow = EscrowPoolClient(
        rpc_url=endpoints.rpc_url, escrow_pool_address=endpoints.escrow_pool,
        ftns_token_address=endpoints.ftns_token, private_key=None)  # read-only

    overall_ok = True
    for nid, rec in state["committed"].items():
        committer = Web3.to_checksum_address(rec["committer_address"])
        key = addr_to_key.get(committer)
        if key is None:
            print(f"ERROR: no node key supplied for committer {committer} (node {nid}); "
                  f"PRSM_NODE_KEYS must include every committing node's key.",
                  file=sys.stderr)
            overall_ok = False
            continue
        contract = Web3SettlementContractClient(
            rpc_url=endpoints.rpc_url,
            contract_address=state.get("registry") or endpoints.settlement_registry,
            private_key=key)
        batch_id = bytes.fromhex(rec["batch_id"])
        report = await run_finalize_by_batch_id(
            contract_client=contract, escrow_client=escrow, batch_id=batch_id)
        print(f"node {committer} batch {batch_id.hex()[:12]}…: {report}")
        if not report.get("success"):
            overall_ok = False
            secs = report.get("seconds_until_finalizable")
            if report.get("finalizable") is False and secs is not None:
                print(f"  not finalizable yet (~{secs/3600:.1f}h remaining)")
    return 0 if overall_ok else 1


async def _run_verify(args, endpoints, chain_id) -> int:
    """READ-ONLY: resolve the deployed addresses + read the committed batch state.
    Safe anytime, NO keys. Confirms the script resolves the live Sepolia
    deployment and reports per-batch status + the expected/actual payouts."""
    from web3 import Web3
    from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
    from prsm.economy.web3.batch_settlement_contract_client import (
        Web3SettlementContractClient,
    )

    print(f"resolved addresses (network={endpoints.network_name} chainId={chain_id}):")
    print(f"  settlement_registry={endpoints.settlement_registry}")
    print(f"  escrow_pool={endpoints.escrow_pool}")
    print(f"  ftns_token={endpoints.ftns_token}")

    state = _load_state(args.state_file)
    if not state or not state.get("committed"):
        print("no phase-1 state with committed share-batches yet; nothing to read "
              "on-chain. (This still confirmed address resolution above.)")
        return 0

    escrow = EscrowPoolClient(
        rpc_url=endpoints.rpc_url, escrow_pool_address=endpoints.escrow_pool,
        ftns_token_address=endpoints.ftns_token, private_key=None)
    contract = Web3SettlementContractClient(
        rpc_url=endpoints.rpc_url,
        contract_address=state.get("registry") or endpoints.settlement_registry,
        private_key=None)

    requester = state.get("requester_address")
    escrow_balance = (
        int(await escrow.balance_of(requester)) if requester else None)
    if requester:
        print(f"requester escrow balance: {escrow_balance} wei")
    total_share = 0
    bench_known_claims = 0  # sum of on-chain PENDING values of the bench-tracked batches
    all_final = True
    for nid, rec in state["committed"].items():
        committer = Web3.to_checksum_address(rec["committer_address"])
        batch_id = bytes.fromhex(rec["batch_id"])
        status = int(await contract.get_batch_status(batch_id))
        finalizable = bool(await contract.is_finalizable(batch_id)) if status == 1 else False
        wallet = int(await escrow.ftns_balance_of(committer))
        # The on-chain PENDING value still drawing on the shared escrow at finalize.
        on_chain_pending = 0
        try:
            b = await contract.get_batch(batch_id)
            if int(b.get("status", status)) == 1:  # PENDING
                on_chain_pending = int(b.get("total_value_ftns", 0))
        except Exception as exc:  # noqa: BLE001
            print(f"    (could not read on-chain batch value: {exc})")
        bench_known_claims += on_chain_pending
        print(f"  node {committer}: batch {batch_id.hex()[:12]}… status={status} "
              f"finalizable={finalizable} expected_share={rec['share_wei']} "
              f"on_chain_pending={on_chain_pending} wallet_ftns={wallet}")
        total_share += int(rec["share_wei"])
        all_final = all_final and (status == 2)
    print(f"sum of expected shares = {total_share} wei (== total {state.get('total_wei')}): "
          f"{total_share == state.get('total_wei')}")
    print(f"all batches FINALIZED: {all_final}")

    # ── sp1161 SHARED-ESCROW SUFFICIENCY DIAGNOSTIC (the bug-catcher) ─────────────
    # Sum the bench-tracked per-stage batches' on-chain PENDING values and compare
    # to the requester's escrow balance. This is the read-only check that would have
    # caught the live under-funding bug immediately.
    if requester and escrow_balance is not None:
        verdict = "SUFFICIENT" if escrow_balance >= bench_known_claims else "SHORT"
        print(f"escrow sufficiency (bench-known): escrow_balance={escrow_balance} wei "
              f"vs bench_known_claims={bench_known_claims} wei => {verdict}")
        print("NOTE: the EscrowPool balance is a SHARED pool per requester. EXTERNAL "
              "pending batches committed OUTSIDE this bench (e.g. a separate two-party "
              "batch) ALSO draw on this SAME escrow and are NOT counted above — the "
              "escrow must cover ALL of the requester's pending batches, not just the "
              "bench-tracked ones. (Enumerating arbitrary external batches needs "
              "event-scanning by a non-indexed requester field and is out of scope; "
              "this NOTE is the honest surface.)")
    return 0


async def _run_deposit_topup(args, endpoints, chain_id) -> int:
    """sp1161 — STANDALONE additive top-up deposit by the requester. A clean
    operator surface to add escrow (e.g. to cover a pre-existing pending batch
    that shares the requester's escrow) instead of an ad-hoc one-liner. Key-gated
    (PRSM_REQUESTER_KEY); echoes the requester ADDRESS + the deposited amount + the
    resulting escrow balance — NEVER the key. Always deposits ADDITIVELY (it never
    skips): this is purpose-built to add escrow on top of whatever is there."""
    from eth_account import Account
    from web3 import Web3
    from prsm.economy.web3.escrow_pool_client import EscrowPoolClient

    requester_key = read_requester_key()  # SystemExit(2) if unset (key-gated)
    requester_addr = Web3.to_checksum_address(Account.from_key(requester_key).address)
    print(f"requester(A)={requester_addr}")  # ADDRESS, never the key

    topup_wei = _amount_to_wei(args.amount_ftns)
    if topup_wei <= 0:
        print("ERROR: --amount-ftns must be positive", file=sys.stderr)
        return 2

    escrow = EscrowPoolClient(
        rpc_url=endpoints.rpc_url, escrow_pool_address=endpoints.escrow_pool,
        ftns_token_address=endpoints.ftns_token, private_key=requester_key)
    pre_balance = int(await escrow.balance_of(requester_addr))
    target = pre_balance + topup_wei
    print(f"depositing {topup_wei} wei additively (pre={pre_balance} -> target>={target}) …")
    deposit_tx = await escrow.deposit(topup_wei)
    print(f"  deposit_tx={deposit_tx}")
    escrow_balance = pre_balance
    for _ in range(6):
        escrow_balance = int(await escrow.balance_of(requester_addr))
        if escrow_balance >= target:
            break
        await asyncio.sleep(args.poll_interval_s)
    print(f"requester {requester_addr} deposited {topup_wei} wei; "
          f"escrow_balance={escrow_balance} wei")
    if escrow_balance < target:
        print(f"WARNING: escrow balance {escrow_balance} < target {target} even after "
              f"polling past RPC lag; check the deposit tx on-chain.", file=sys.stderr)
        return 1
    return 0


async def _amain(args) -> int:
    from prsm.config.networks import resolve_endpoints
    from web3 import Web3

    network = (os.environ.get("PRSM_NETWORK", "") or "testnet").strip() or "testnet"
    endpoints = resolve_endpoints(network=network)
    for field in ("rpc_url", "escrow_pool", "ftns_token", "settlement_registry"):
        if not getattr(endpoints, field, None):
            print(f"ERROR: {field} not configured for network '{network}'",
                  file=sys.stderr)
            return 2

    # Authoritative chain id from the LIVE RPC; hard-guard mainnet (identical to
    # settlement_sepolia_e2e.py — refuse 8453 without --i-understand-mainnet).
    w3 = Web3(Web3.HTTPProvider(endpoints.rpc_url))
    chain_id = w3.eth.chain_id
    assert_chain_allowed(chain_id=chain_id, mainnet_acknowledged=args.i_understand_mainnet)
    from urllib.parse import urlsplit
    _u = urlsplit(endpoints.rpc_url)
    print(f"network={network} chainId={chain_id} rpc={_u.scheme}://{_u.netloc}")

    if args.phase == "deposit-commit":
        return await _run_phase1(args, endpoints, chain_id)
    if args.phase == "finalize":
        return await _run_phase2(args, endpoints, chain_id)
    if args.phase == "verify":
        return await _run_verify(args, endpoints, chain_id)
    if args.phase == "deposit-topup":
        return await _run_deposit_topup(args, endpoints, chain_id)
    print(f"ERROR: unknown phase {args.phase!r}", file=sys.stderr)
    return 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase",
                        choices=("deposit-commit", "finalize", "verify", "deposit-topup"),
                        default="verify",
                        help="deposit-commit (phase 1: deposit + per-node commit), "
                             "finalize (phase 2: per-node finalize after the window), "
                             "verify (read-only; resolves addresses + reads batch "
                             "state + reports escrow-vs-claims sufficiency; NO keys), "
                             "or deposit-topup (standalone additive requester deposit "
                             "of --amount-ftns). Default: verify (the safe default).")
    parser.add_argument("--amount-ftns", type=float, default=1.0,
                        help="[deposit-commit] TOTAL FTNS the requester deposits + the "
                             "nodes split; [deposit-topup] FTNS to add to escrow "
                             "(default 1.0)")
    parser.add_argument("--assume-dedicated-escrow", action="store_true",
                        help="[deposit-commit] DANGER: skip the deposit when the escrow "
                             "already holds >= --amount-ftns (old behavior). ONLY safe "
                             "if this requester's escrow is DEDICATED to this per-stage "
                             "run. By DEFAULT the bench deposits the per-stage total "
                             "ADDITIVELY because EscrowPool balance is a SHARED pool "
                             "that may back OTHER pending batches — skipping would "
                             "silently under-fund and the finalize would revert.")
    parser.add_argument("--state-file", default=_default_state_file(),
                        help="durable state file bridging the two phases (the committed "
                             "batch_ids + committer addresses)")
    parser.add_argument("--poll-interval-s", type=float, default=15.0,
                        help="seconds between escrow-balance lag polls")
    parser.add_argument("--i-understand-mainnet", action="store_true",
                        help="DANGER: allow Base mainnet (chainId 8453). Sends real "
                             "transactions. Default + intended target is Base Sepolia.")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
