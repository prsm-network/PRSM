#!/usr/bin/env python3
"""Multi-stage settlement mainnet-activation canary driver (requester side).

Automates runbook §5.1-5.3 (quote → sign the per-stage auth → paid multi-stage inference) so the
operator runs ONE command for the canary instead of manual curl + Python. Requester-side only: it
POSTs the auth + inference; the STAGE NODES then self-commit their own share-batches on-chain (their
keys, their signatures) via their settlement poll loop. Verify per runbook §6 afterward.

Env (keys live ONLY in your shell — never argv/chat):
  REQUESTER_KEY   the requester's key — signs the per-stage authorization + owns the on-chain escrow.
  HEAD_URL        the head (stage-0) node base URL, e.g. https://nodeA:8000  (default localhost:8000).
  MODEL           model id (default a big instruct model; a tiny sharded model also works).
  PROMPT, MAX_TOKENS, BUDGET_FTNS   canary knobs — keep BUDGET_FTNS tiny (default 0.02).

Usage:
  REQUESTER_KEY=0x… HEAD_URL=https://nodeA:8000 BUDGET_FTNS=0.02 \
    python3 scripts/multistage_settlement_canary.py
"""
from __future__ import annotations

import os
import sys
import time
from decimal import Decimal

REQ = (os.environ.get("REQUESTER_KEY") or "").strip()
HEAD = (os.environ.get("HEAD_URL") or "http://localhost:8000").strip().rstrip("/")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
PROMPT = os.environ.get("PROMPT", "The capital of France is")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8"))
BUDGET = float(os.environ.get("BUDGET_FTNS", "0.02"))


def main() -> int:
    if not REQ:
        print("Set REQUESTER_KEY (owns the escrow + signs the per-stage auth).", file=sys.stderr)
        return 2

    import httpx

    from prsm.settlement.payment_client import build_per_stage_payment_authorization

    print(f"head: {HEAD}   model: {MODEL}   budget: {BUDGET} FTNS   max_tokens: {MAX_TOKENS}")

    print("\n[1/3] quote-multistage — preview the payee set…")
    with httpx.Client(timeout=120.0) as c:
        q = c.post(f"{HEAD}/compute/inference/quote-multistage",
                   json={"model_id": MODEL, "prompt": PROMPT, "max_tokens": MAX_TOKENS,
                         "budget_ftns": BUDGET})
    if q.status_code != 200:
        print(f"  quote failed ({q.status_code}): {q.text[:300]}", file=sys.stderr)
        return 1
    quote = q.json()
    if not quote.get("settleable"):
        print(f"  NOT settleable (multi_stage={quote.get('multi_stage')}) — check the wallet map "
              f"(runbook §2): {quote}", file=sys.stderr)
        return 1
    print(f"  stage_count={quote['stage_count']}  payees={quote['payees']}")
    print(f"  payee_set_hash={quote['payee_set_hash']}  total_value_wei={quote['total_value_wei']}")

    print("[2/3] sign the per-stage authorization over the QUOTED payee set…")
    payees_ftns = [(addr, Decimal(str(share)) / Decimal(10 ** 18))
                   for addr, share in quote["payees"]]
    auth = build_per_stage_payment_authorization(
        requester_key=REQ, payees=payees_ftns, model_id=MODEL, prompt=PROMPT,
        max_tokens=MAX_TOKENS, privacy_tier="none", content_tier="A",
        expiry_unix=int(time.time()) + 3600)
    if auth["payload"]["payee_set_hash"] != quote["payee_set_hash"]:
        print("  payee_set_hash != quote — topology drift; re-quote/re-sign.", file=sys.stderr)
        return 1
    print("  auth bound to the quoted payee set ✓")

    print("[3/3] paid multi-stage inference (head serves + splits + delivers per-stage tasks)…")
    with httpx.Client(timeout=300.0) as c:
        r = c.post(f"{HEAD}/compute/inference",
                   json={"model_id": MODEL, "prompt": PROMPT, "max_tokens": MAX_TOKENS,
                         "per_stage_payment_authorization": auth})
    if r.status_code != 200:
        print(f"  inference failed ({r.status_code}): {r.text[:300]}", file=sys.stderr)
        return 1
    out = r.json()
    print(f"  output: {(out.get('output') or out.get('text'))!r}")

    print("\n✅ CANARY DISPATCHED. Now, on EACH stage node:")
    print("   - the settlement poll loop logs 'per-stage commit cycle: committed 1/1' → capture the")
    print("     batchId + commit tx (msg.sender == that node's settler).")
    print("   - after the challenge window, finalize → EscrowPool draws the requester's escrow.")
    print("   Verify per runbook §6: conservation (shares sum to total), self-commit, escrow draw.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
