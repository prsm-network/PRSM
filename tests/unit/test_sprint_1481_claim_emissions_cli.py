"""Sprint 1481 — the claim surface: manifest verification + `prsm node claim-emissions`.

Merkle PROOFS are not on chain (only the root is), so an operator's (amount, proof)
must come from an off-chain epoch manifest. That manifest is UNTRUSTED input — a
stale, wrong-deployment, or hostile one can hand an operator a proof for the wrong
amount or an epoch that was never published.

The property under test: the client re-derives the leaf and folds the proof against
the root READ FROM THE CONTRACT, never against the root the manifest asserts, and
REFUSES rather than spending gas when they disagree. (The contract is safe either
way — it verifies the proof itself and pays the leaf's account, not the sender — so
the worst a bad manifest can do is waste gas. Failing locally is cheaper and tells
the operator exactly what is wrong.)
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from prsm.economy.web3.operator_reward_pool_client import (
    ManifestMismatchError,
    OperatorRewardPoolClient,
    manifest_entry_for,
    verify_manifest_entry_against_chain,
)
from prsm.settlement.reward_epoch import RewardEntry, build_reward_epoch

ALICE = "0x1111111111111111111111111111111111111111"
BOB = "0x2222222222222222222222222222222222222222"
CAROL = "0x3333333333333333333333333333333333333333"
EPOCH_ID = 11


def _epoch_and_manifest():
    entries = [
        RewardEntry(ALICE, 100 * 10**18),
        RewardEntry(BOB, 250 * 10**18),
        RewardEntry(CAROL, 50 * 10**18),
    ]
    ep = build_reward_epoch(EPOCH_ID, entries)
    manifest = {
        "epoch_id": EPOCH_ID,
        "merkle_root": ep.root_hex,
        "total_amount_wei": str(ep.total_amount_wei),
        "entries": [
            {"account": e.account, "amount_wei": str(e.amount_wei),
             "proof": ep.proof_hex(e.account)}
            for e in ep.entries
        ],
    }
    return ep, manifest


class _StubClient:
    """Duck-typed stand-in so the REAL resolve_claimable logic is exercised
    without web3. Only the two chain reads it uses are stubbed."""

    def __init__(self, root: bytes, published: bool = True, claimed: bool = False,
                 head_block: int = 1234):
        self._root, self._published, self._claimed = root, published, claimed
        # sp1481 — resolve_claimable pins both reads to ONE height, so the stub
        # must expose a head like the real client does.
        self.seen_block_tags = []
        self.web3 = SimpleNamespace(eth=SimpleNamespace(block_number=head_block))

    def get_epoch(self, epoch_id, block_tag=None):
        self.seen_block_tags.append(("get_epoch", block_tag))
        return {
            "epoch_id": epoch_id,
            "merkle_root": self._root,
            "total_amount_wei": 0,
            "claimed_amount_wei": 0,
            "published_at": 1 if self._published else 0,
            "reclaimed": False,
            "published": self._published,
        }

    def has_claimed(self, epoch_id, account, block_tag=None):
        self.seen_block_tags.append(("has_claimed", block_tag))
        return self._claimed

    def resolve_claimable(self, manifest, account):
        # Delegate to the REAL implementation so the CLI tests exercise production
        # verification logic rather than a mock that could diverge from it.
        return OperatorRewardPoolClient.resolve_claimable(self, manifest, account)


def _resolve(stub, manifest, account):
    # Unbound call: run the production method against the stub.
    return OperatorRewardPoolClient.resolve_claimable(stub, manifest, account)


# ───────────────────────── manifest verification ─────────────────────────

def test_valid_manifest_verifies_against_on_chain_root():
    ep, manifest = _epoch_and_manifest()
    got = _resolve(_StubClient(ep.merkle_root), manifest, ALICE)
    assert got is not None
    assert got.amount_wei == 100 * 10**18
    assert got.already_claimed is False


def test_tampered_amount_is_refused():
    """★ A manifest inflating an operator's amount must be caught locally."""
    ep, manifest = _epoch_and_manifest()
    manifest["entries"][0]["amount_wei"] = str(999 * 10**18)
    with pytest.raises(ManifestMismatchError, match="does NOT verify"):
        _resolve(_StubClient(ep.merkle_root), manifest, manifest["entries"][0]["account"])


def test_manifest_lying_about_its_own_root_is_refused():
    """★ THE key property: verification uses the CHAIN's root, so a manifest that
    is internally consistent with a FORGED root still fails. If we validated
    against manifest['merkle_root'], this forged epoch would sail through."""
    forged_entries = [RewardEntry(ALICE, 10_000 * 10**18)]
    forged = build_reward_epoch(EPOCH_ID, forged_entries)
    manifest = {
        "epoch_id": EPOCH_ID,
        "merkle_root": forged.root_hex,          # self-consistent...
        "entries": [{"account": ALICE, "amount_wei": str(10_000 * 10**18),
                     "proof": forged.proof_hex(ALICE)}],
    }
    real, _ = _epoch_and_manifest()              # ...but not what the chain published
    with pytest.raises(ManifestMismatchError, match="does NOT verify"):
        _resolve(_StubClient(real.merkle_root), manifest, ALICE)


def test_proof_for_another_account_is_refused():
    ep, manifest = _epoch_and_manifest()
    manifest["entries"][0]["account"] = CAROL   # Carol's address, Alice's proof/amount
    with pytest.raises(ManifestMismatchError):
        _resolve(_StubClient(ep.merkle_root), manifest, CAROL)


def test_unpublished_epoch_is_refused():
    ep, manifest = _epoch_and_manifest()
    with pytest.raises(ManifestMismatchError, match="not published"):
        _resolve(_StubClient(ep.merkle_root, published=False), manifest, ALICE)


def test_no_entry_for_account_returns_none():
    ep, manifest = _epoch_and_manifest()
    assert _resolve(_StubClient(ep.merkle_root), manifest, "0x" + "9" * 40) is None


def test_already_claimed_is_surfaced_not_resubmitted():
    ep, manifest = _epoch_and_manifest()
    got = _resolve(_StubClient(ep.merkle_root, claimed=True), manifest, ALICE)
    assert got is not None and got.already_claimed is True


def test_manifest_lookup_is_case_insensitive():
    _, manifest = _epoch_and_manifest()
    assert manifest_entry_for(manifest, ALICE.upper().replace("0X", "0x")) is not None


def test_verify_helper_uses_supplied_chain_root():
    ep, manifest = _epoch_and_manifest()
    e = manifest["entries"][0]
    proof = [bytes.fromhex(p[2:]) for p in e["proof"]]
    assert verify_manifest_entry_against_chain(
        epoch_id=EPOCH_ID, account=e["account"], amount_wei=int(e["amount_wei"]),
        proof=proof, on_chain_root=ep.merkle_root)
    assert not verify_manifest_entry_against_chain(
        epoch_id=EPOCH_ID, account=e["account"], amount_wei=int(e["amount_wei"]),
        proof=proof, on_chain_root=b"\x00" * 32)


def test_both_chain_reads_are_pinned_to_the_same_block():
    """★ Live Base Sepolia finding: reading the epoch record and the claimed flag
    separately at "latest" lets load-balanced replicas answer from DIFFERENT
    heights — re-running right after a successful claim reported "claimable"
    because hasClaimed was still false on a lagging replica, which would send the
    operator into a tx that reverts AlreadyClaimed. Same mixed-height class as the
    sp1474 reconciler bug. Both reads must be pinned to ONE height."""
    ep, manifest = _epoch_and_manifest()
    stub = _StubClient(ep.merkle_root, head_block=999)
    got = _resolve(stub, manifest, ALICE)
    assert got is not None
    tags = [t for _name, t in stub.seen_block_tags]
    assert tags, "no chain reads recorded"
    assert all(t == 999 for t in tags), f"reads at mixed heights: {stub.seen_block_tags}"
    assert got.read_at_block == 999, "the height the answer reflects must be surfaced"


# ───────────────────────── CLI behaviour ─────────────────────────

def _run_cli(tmp_path, monkeypatch, manifest, stub, args=(), env=None):
    from prsm.cli import node
    import prsm.economy.web3.operator_reward_pool_client as mod

    mpath = tmp_path / "epoch.json"
    mpath.write_text(json.dumps(manifest))

    monkeypatch.setattr(mod, "OperatorRewardPoolClient",
                        lambda **kw: stub, raising=True)

    class _Ep:
        network_name, chain_id, rpc_url = "base-sepolia", 84532, "http://stub"
    import prsm.config.networks as nets
    monkeypatch.setattr(nets, "resolve_endpoints", lambda *a, **k: _Ep(), raising=True)

    base_env = {
        "PRSM_OPERATOR_ADDRESS": ALICE,
        "PRSM_REWARD_POOL_ADDRESS": "0x" + "5" * 40,
        "FTNS_WALLET_PRIVATE_KEY": "0x" + "11" * 32,
    }
    base_env.update(env or {})
    return CliRunner().invoke(
        node, ["claim-emissions", "--manifest", str(mpath), *args],
        env=base_env, catch_exceptions=False,
    )


def test_cli_dry_run_reports_claimable(tmp_path, monkeypatch):
    ep, manifest = _epoch_and_manifest()
    res = _run_cli(tmp_path, monkeypatch, manifest, _StubClient(ep.merkle_root),
                   args=("--dry-run", "--format", "json"))
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["claimable"] is True
    assert payload["verified_against_chain"] is True
    assert payload["amount_wei"] == str(100 * 10**18)


def test_cli_refuses_mismatched_manifest_with_exit_3(tmp_path, monkeypatch):
    """★ The operator-visible half of the security property: a bad manifest exits
    3 (refused) and NEVER reaches the claim path."""
    ep, manifest = _epoch_and_manifest()
    manifest["entries"][0]["amount_wei"] = str(999 * 10**18)

    class _NoClaim(_StubClient):
        def claim(self, *a, **k):  # pragma: no cover - must never run
            raise AssertionError("claim() must not be reached on a bad manifest")

    res = _run_cli(tmp_path, monkeypatch, manifest, _NoClaim(ep.merkle_root),
                   args=("--format", "json"))
    assert res.exit_code == 3, res.output
    assert json.loads(res.output)["error"] == "manifest_mismatch"


def test_cli_no_entitlement_exits_1(tmp_path, monkeypatch):
    ep, manifest = _epoch_and_manifest()
    res = _run_cli(tmp_path, monkeypatch, manifest, _StubClient(ep.merkle_root),
                   args=("--dry-run", "--format", "json"),
                   env={"PRSM_OPERATOR_ADDRESS": "0x" + "9" * 40})
    assert res.exit_code == 1, res.output
    assert json.loads(res.output)["reason"] == "no_entry"


def test_cli_already_claimed_exits_1(tmp_path, monkeypatch):
    ep, manifest = _epoch_and_manifest()
    res = _run_cli(tmp_path, monkeypatch, manifest,
                   _StubClient(ep.merkle_root, claimed=True),
                   args=("--dry-run", "--format", "json"))
    assert res.exit_code == 1, res.output
    assert json.loads(res.output)["reason"] == "already_claimed"


def test_cli_claim_sends_and_reports_tx(tmp_path, monkeypatch):
    ep, manifest = _epoch_and_manifest()
    sent = {}

    class _Claiming(_StubClient):
        def claim(self, epoch_id, account, amount_wei, proof):
            sent.update(epoch_id=epoch_id, account=account, amount_wei=amount_wei)
            return "0x" + "ab" * 32

    res = _run_cli(tmp_path, monkeypatch, manifest, _Claiming(ep.merkle_root),
                   args=("--format", "json"))
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["claimed"] is True
    assert payload["tx_hash"] == "0x" + "ab" * 32
    # ★ The claim is made FOR the earner's address, not the signer's.
    assert sent["account"].lower() == ALICE.lower()
    assert sent["amount_wei"] == 100 * 10**18


def test_cli_requires_a_signer_to_actually_claim(tmp_path, monkeypatch):
    ep, manifest = _epoch_and_manifest()
    res = _run_cli(tmp_path, monkeypatch, manifest, _StubClient(ep.merkle_root),
                   args=("--format", "json"),
                   env={"FTNS_WALLET_PRIVATE_KEY": ""})
    assert res.exit_code == 2, res.output
    assert json.loads(res.output)["error"] == "no_key"


def test_cli_dry_run_works_without_a_signer(tmp_path, monkeypatch):
    """A read-only operator must be able to check entitlements with no key."""
    ep, manifest = _epoch_and_manifest()
    res = _run_cli(tmp_path, monkeypatch, manifest, _StubClient(ep.merkle_root),
                   args=("--dry-run", "--format", "json"),
                   env={"FTNS_WALLET_PRIVATE_KEY": ""})
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["claimable"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
