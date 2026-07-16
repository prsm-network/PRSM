"""Sprint 1465 — `prsm marketplace sign-stake-binding` CLI.

Closes the operator loop for the sp1459-activated marketplace advertiser: to be weighted by REAL
bonded stake (sp1457) instead of the self-declared tier label, an advertising node must set
PRSM_STAKE_ETH_ADDRESS + PRSM_STAKE_BINDING_SIG (design doc §14). This command produces that pair by
signing build_stake_binding_message(provider_id, address) with the operator's stake eth key — read
from the environment, NEVER passed on the command line (keys must not land in argv/shell history).
"""
from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner


def test_cli_sign_stake_binding_requires_env_key(monkeypatch):
    from prsm.cli import main
    monkeypatch.delenv("PRSM_STAKE_ETH_KEY", raising=False)
    r = CliRunner().invoke(main, ["marketplace", "sign-stake-binding"])
    assert r.exit_code == 1
    assert "PRSM_STAKE_ETH_KEY" in r.output          # tells the operator to use the env var


def test_cli_sign_stake_binding_emits_verifiable_binding(monkeypatch):
    from eth_account import Account
    from prsm.cli import main
    from prsm.marketplace.listing import verify_stake_binding

    key = "0x" + "5a" * 32
    node_id = "a" * 32
    monkeypatch.setenv("PRSM_STAKE_ETH_KEY", key)
    monkeypatch.setattr("prsm.node.identity.load_node_identity",
                        lambda *a, **k: SimpleNamespace(node_id=node_id))
    monkeypatch.setattr("prsm.node.config.NodeConfig.load",
                        classmethod(lambda cls: SimpleNamespace(identity_path="ignored")))

    r = CliRunner().invoke(main, ["marketplace", "sign-stake-binding"])
    assert r.exit_code == 0, r.output

    def _val(prefix):
        line = [l for l in r.output.splitlines() if prefix in l][0]
        return line.split("=", 1)[1].strip()

    address = _val("PRSM_STAKE_ETH_ADDRESS=")
    sig = _val("PRSM_STAKE_BINDING_SIG=")
    assert address == Account.from_key(key).address       # binding is to the key's OWN address
    # ★ the emitted binding authenticates for THIS node's provider_id → the selector will honor it.
    assert verify_stake_binding(node_id, address, sig) is True


def test_cli_sign_stake_binding_never_takes_key_on_argv(monkeypatch):
    # Defensive: passing the key as an argument must NOT be how it's supplied (no such option/arg).
    from prsm.cli import main
    monkeypatch.delenv("PRSM_STAKE_ETH_KEY", raising=False)
    r = CliRunner().invoke(main, ["marketplace", "sign-stake-binding", "0x" + "11" * 32])
    assert r.exit_code != 0                               # extra argv arg rejected (no positional key)


def test_cli_sign_stake_binding_json_format(monkeypatch):
    from prsm.cli import main
    from prsm.marketplace.listing import verify_stake_binding

    monkeypatch.setenv("PRSM_STAKE_ETH_KEY", "0x" + "5a" * 32)
    node_id = "b" * 32
    monkeypatch.setattr("prsm.node.identity.load_node_identity",
                        lambda *a, **k: SimpleNamespace(node_id=node_id))
    monkeypatch.setattr("prsm.node.config.NodeConfig.load",
                        classmethod(lambda cls: SimpleNamespace(identity_path="ignored")))
    r = CliRunner().invoke(main, ["marketplace", "sign-stake-binding", "--format", "json"])
    assert r.exit_code == 0, r.output
    import json
    payload = json.loads(r.output.strip().splitlines()[-1])
    assert payload["provider_id"] == node_id
    assert verify_stake_binding(
        node_id, payload["stake_eth_address"], payload["stake_binding_sig"]) is True
