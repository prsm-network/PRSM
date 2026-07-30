"""Sprint 1487 — `prsm node publish-epoch` exit-code contract.

This command spends the emission pot. Its exit codes are the operator-visible
contract an automated runner keys on, so a safety refusal returning 0 would let a
cron job treat "I declined to double-pay" as "published successfully" and move on.

  0 — published (or a plan was produced)
  1 — nothing to pay right now
  2 — misconfigured
  3 — REFUSED for a safety reason
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node

POOL = "0x" + "c3" * 20
REG = "0x" + "d4" * 20
BASE_ENV = {
    "PRSM_REWARD_POOL_ADDRESS": POOL,
    "PRSM_BATCH_REGISTRY_ADDRESS": REG,
    "FTNS_WALLET_PRIVATE_KEY": "0x" + "11" * 32,
}


def _run(args, env=None, **patches):
    runner = CliRunner()
    e = dict(BASE_ENV)
    if env is not None:
        e = env
    with patch("prsm.config.networks.resolve_endpoints") as rp, \
         patch("prsm.economy.web3.operator_reward_pool_client.OperatorRewardPoolClient") as pc, \
         patch("prsm.economy.web3.batch_settlement_contract_client."
               "Web3SettlementContractClient") as rc:
        rp.return_value = MagicMock(rpc_url="http://x", chain_id=8453)
        rc.return_value.scan_finalized_batches.return_value = patches.get("batches", [])
        pool = pc.return_value
        pool.epoch_exists.side_effect = patches.get("epoch_exists", lambda _i: False)
        pool.unreserved_balance_wei.return_value = patches.get("unreserved", 10**18)
        pool.publish_epoch.return_value = "0xdeadbeef"
        return runner.invoke(node, ["publish-epoch"] + args, env=e)


def _batch(bid, provider, value=100, ts=1000):
    from prsm.settlement.emission_epoch import FinalizedBatch
    return FinalizedBatch(bid, provider, value, ts)


A = "0x" + "a1" * 20


# ── misconfiguration (2) ────────────────────────────────────────────

def test_missing_pool_address_exits_2():
    r = _run([], env={"PRSM_BATCH_REGISTRY_ADDRESS": REG})
    assert r.exit_code == 2 and "OperatorRewardPool address" in r.output


def test_missing_registry_address_exits_2():
    r = _run([], env={"PRSM_REWARD_POOL_ADDRESS": POOL})
    assert r.exit_code == 2 and "BatchSettlementRegistry address" in r.output


def test_execute_without_a_publisher_key_exits_2(tmp_path):
    r = _run(["--execute", "--watermark", str(tmp_path / "wm.json")],
             env={"PRSM_REWARD_POOL_ADDRESS": POOL,
                  "PRSM_BATCH_REGISTRY_ADDRESS": REG})
    assert r.exit_code == 2 and "rootPublisher key" in r.output


# ── the safety refusal must NOT look like success ───────────────────

def test_lost_watermark_exits_3_not_0(tmp_path):
    """★ A cron runner keys on the exit code. Returning 0 here would let it treat
    'I refused to double-pay' as 'published' and carry on."""
    r = _run(["--execute", "--watermark", str(tmp_path / "wm.json"),
              "--manifest-dir", str(tmp_path)],
             batches=[_batch("0xaa", A)],
             epoch_exists=lambda i: i == 1)          # chain has epoch 1; wm is cold
    assert r.exit_code == 3
    assert "REFUSED" in r.output and "lost, not cold" in r.output


def test_corrupt_watermark_exits_3(tmp_path):
    p = tmp_path / "wm.json"
    p.write_text("{not json")
    r = _run(["--watermark", str(p), "--manifest-dir", str(tmp_path)],
             batches=[_batch("0xaa", A)])
    assert r.exit_code == 3 and "REFUSING" in r.output


def test_pot_over_the_unreserved_balance_exits_3(tmp_path):
    r = _run(["--execute", "--pot", "5", "--watermark", str(tmp_path / "wm.json"),
              "--manifest-dir", str(tmp_path)],
             batches=[_batch("0xaa", A)], unreserved=10**18)   # 1 FTNS available
    assert r.exit_code == 3 and "UNRESERVED" in r.output


# ── nothing to pay (1) ──────────────────────────────────────────────

def test_no_eligible_batches_exits_1(tmp_path):
    r = _run(["--watermark", str(tmp_path / "wm.json"),
              "--manifest-dir", str(tmp_path)], batches=[])
    assert r.exit_code == 1 and "Nothing to publish" in r.output


# ── plan vs publish (0) ─────────────────────────────────────────────

def test_defaults_to_planning_and_publishes_nothing(tmp_path):
    """★ Publishing is irreversible and an epoch id can never be rewritten, so the
    bare command must never broadcast."""
    r = _run(["--watermark", str(tmp_path / "wm.json"),
              "--manifest-dir", str(tmp_path)], batches=[_batch("0xaa", A)])
    assert r.exit_code == 0
    assert "PLAN (nothing published)" in r.output
    assert "--execute to publish" in r.output
    assert not (tmp_path / "wm.json").exists()      # watermark untouched


def test_execute_publishes_and_reports_the_tx(tmp_path):
    r = _run(["--execute", "--watermark", str(tmp_path / "wm.json"),
              "--manifest-dir", str(tmp_path)],
             batches=[_batch("0xaa", A), _batch("0xbb", "0x" + "b2" * 20)])
    assert r.exit_code == 0
    assert "✅ Published" in r.output and "0xdeadbeef" in r.output
    assert (tmp_path / "wm.json").exists()          # watermark advanced
    assert (tmp_path / "epoch-1.json").exists()     # manifest written


def test_json_output_carries_the_fields_a_runner_needs(tmp_path):
    r = _run(["--watermark", str(tmp_path / "wm.json"), "--manifest-dir",
              str(tmp_path), "--format", "json"], batches=[_batch("0xaa", A)])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["dry_run"] is True and data["published"] is False
    assert data["epoch_id"] == 1 and data["recipients"] == 1
    assert data["merkle_root"].startswith("0x")
    assert data["watermark"].endswith("wm.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
