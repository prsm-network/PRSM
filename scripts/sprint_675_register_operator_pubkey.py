"""Sprint 675 — register any new operator's pubkey on the live
PublisherKeyAnchor.

Generalized variant of:
  - sprint 623 (hardcoded mac + droplet)
  - sprint 674 (Lambda-specific naming)

Use for:
  - 2nd DigitalOcean droplet (sprint 675 deploy playbook)
  - Lambda GPU operator (sprint 674 deploy playbook)
  - Any future operator joining the live fleet

Required env vars:
  PRSM_DEPLOYER_PRIVATE_KEY  0x-prefixed Base EOA private key
                             (funded with ~0.0001 ETH for gas)
  OPERATOR_NODE_ID           node_id from the new operator's
                             ~/.prsm/identity.json (32 hex chars)
  OPERATOR_PUBKEY_B64        base64 pubkey from the same identity.json

Sends ONE register(OPERATOR_PUBKEY) TX to the anchor at
0xd811ad9986f44f404b0fd992168a7cc76206df03 on Base mainnet. Gas
~50k → ~$0.01.

Idempotent: skips re-registration if anchor.lookup() already
shows a pubkey for this node_id.

NEVER COMMIT THE DEPLOYER PRIVATE KEY. Export via env at run time.
"""
from __future__ import annotations

import base64
import os
import sys
import time

# Sprint 1029 — make the script importable however it is invoked. Running it as
# a file (`python scripts/sprint_675_...py`) puts scripts/ on sys.path[0], not
# the repo root, and the editable install does not resolve `prsm` in script-mode
# on every interpreter (observed on CPython 3.14). Prepend the repo root so the
# deferred `from prsm.security...` import in main() always resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


ANCHOR = "0xd811ad9986f44f404b0fd992168a7cc76206df03"
RPC = os.environ.get("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
CHAIN_ID = 8453


def main() -> int:
    pk = (os.environ.get("PRSM_DEPLOYER_PRIVATE_KEY", "") or "").strip()
    if not pk:
        print(
            "ERROR: PRSM_DEPLOYER_PRIVATE_KEY env not set. "
            "Export your funded Base EOA private key (0x-prefixed).",
            file=sys.stderr,
        )
        return 1
    if not pk.startswith("0x"):
        pk = "0x" + pk

    node_id = (
        os.environ.get("OPERATOR_NODE_ID", "") or ""
    ).strip().lower()
    if len(node_id) != 32 or not all(
        c in "0123456789abcdef" for c in node_id
    ):
        print(
            f"ERROR: OPERATOR_NODE_ID must be 32 lowercase hex chars; "
            f"got {node_id!r}",
            file=sys.stderr,
        )
        return 1

    pubkey_b64 = (
        os.environ.get("OPERATOR_PUBKEY_B64", "") or ""
    ).strip()
    if not pubkey_b64:
        print(
            "ERROR: OPERATOR_PUBKEY_B64 env not set. Copy from the "
            "new operator's ~/.prsm/identity.json `public_key_b64` field.",
            file=sys.stderr,
        )
        return 1
    try:
        pubkey_bytes = base64.b64decode(pubkey_b64)
    except Exception as exc:
        print(f"ERROR: OPERATOR_PUBKEY_B64 decode failed: {exc}", file=sys.stderr)
        return 1
    if len(pubkey_bytes) != 32:
        print(
            f"ERROR: pubkey must decode to 32 bytes (Ed25519); "
            f"got {len(pubkey_bytes)} bytes",
            file=sys.stderr,
        )
        return 1

    from web3 import Web3
    from eth_account import Account

    w3 = Web3(Web3.HTTPProvider(RPC))
    if not w3.is_connected():
        print(f"ERROR: Web3 not connected to {RPC}", file=sys.stderr)
        return 1

    deployer = Account.from_key(pk)
    print(f"Deployer EOA: {deployer.address}")
    balance = w3.eth.get_balance(deployer.address)
    print(f"Balance: {Web3.from_wei(balance, 'ether'):.6f} ETH")
    if balance < Web3.to_wei(0.0001, "ether"):
        print(
            "ERROR: Deployer balance < 0.0001 ETH — won't cover gas. "
            "Top up before re-running.",
            file=sys.stderr,
        )
        return 1

    from prsm.security.publisher_key_anchor.client import (
        PublisherKeyAnchorClient,
    )
    anchor = PublisherKeyAnchorClient(
        contract_address=ANCHOR, rpc_url=RPC,
    )
    existing = anchor.lookup(node_id)
    if existing:
        print(
            f"⚠ node_id {node_id} already has pubkey "
            f"{existing[:16]}... on anchor. Skipping registration.",
        )
        return 0

    # Sprint 675 fix — actual contract signature is
    # `register(bytes publicKey)`; the contract derives nodeId on-chain
    # via `bytes16(sha256(publicKey))`. Earlier draft of this script
    # used a phantom `register(bytes16, bytes32)` signature → reverted
    # at the dispatcher because the selector doesn't exist on the
    # deployed contract (2026-05-21 attempted TX
    # 0x37ffa34236d843135941e450221ebd849d20b3e47b4ba6c4e0a06c905ee263a8
    # reverted with 23k gas, no logs — classic non-existent-function
    # revert shape).
    register_abi = [{
        "name": "register",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "publicKey", "type": "bytes"}],
        "outputs": [],
    }]
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(ANCHOR),
        abi=register_abi,
    )
    # Defensive: cross-check the supplied node_id matches what the
    # contract will derive. Mismatch = caller bug (sent wrong pair).
    import hashlib as _hashlib
    expected_node_id = _hashlib.sha256(pubkey_bytes).digest()[:16].hex()
    if expected_node_id != node_id:
        print(
            f"ERROR: supplied node_id {node_id} doesn't match "
            f"contract-derived sha256(pubkey)[:16] = {expected_node_id}. "
            f"Check that OPERATOR_NODE_ID + OPERATOR_PUBKEY_B64 come "
            f"from the same identity.json.",
            file=sys.stderr,
        )
        return 1

    nonce = w3.eth.get_transaction_count(deployer.address)
    gas_price = w3.eth.gas_price
    tx = contract.functions.register(pubkey_bytes).build_transaction({
        "from": deployer.address,
        "nonce": nonce,
        # Sprint 676 fix — actual register() execution uses ~85-95k gas
        # (two cold SSTOREs to publisherKeys + registeredAt, plus event
        # emission with 1 indexed bytes16 + 1 unindexed bytes + 1
        # unindexed uint64). Sprint 623's working script used 100k; we
        # use 150k for safety margin. Out-of-gas revert observed
        # 2026-05-21 on TX 0x0d2ffd74... with 80k limit.
        "gas": 150000,
        "gasPrice": gas_price,
        "chainId": CHAIN_ID,
    })
    signed = deployer.sign_transaction(tx)
    raw_tx = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw_tx)
    print(f"TX sent: {tx_hash.hex()}")
    print("Waiting for confirmation...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status != 1:
        print(f"✗ TX reverted. Receipt: {dict(receipt)}", file=sys.stderr)
        return 1
    print(
        f"✓ Confirmed in block {receipt.blockNumber} "
        f"(gas used: {receipt.gasUsed})"
    )

    time.sleep(2)
    looked_up = anchor.lookup(node_id)
    if not looked_up:
        print(
            "✗ anchor.lookup returned empty after confirmation — "
            "RPC indexing delay?",
            file=sys.stderr,
        )
        return 1
    if looked_up != pubkey_b64:
        print(
            f"✗ Looked-up pubkey doesn't match registered:\n"
            f"  expected: {pubkey_b64}\n"
            f"  got:      {looked_up}",
            file=sys.stderr,
        )
        return 1
    print(
        f"🎯 anchor.lookup({node_id}) = {looked_up[:16]}... ✓"
    )
    print()
    print(f"Operator {node_id} is now registered on the live anchor.")
    print(f"This identity can act as a stage in multi-host inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
