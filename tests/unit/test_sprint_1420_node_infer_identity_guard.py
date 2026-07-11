"""Sprint 1420 — `prsm node infer` crashed with a raw AttributeError when there is no identity.

`node_infer_cli` did:

    cfg = NodeConfig.load()
    settler = load_node_identity(cfg.identity_path)      # -> None if ~/.prsm/identity.json absent
    ...
    f"{settler.node_id[:8]}"                             # -> AttributeError: 'NoneType' ...

`settler` signs the §7 receipt, so it is needed even when the daemon is remote. The two SIBLING
call sites of load_node_identity in this CLI (`prsm node info` at ~4460, and the fiat-readiness
command at ~6906) both guard with `if not identity: <clean message>; return`. This one did not.

It is reachable: the reachability check ahead of it only proves THE DAEMON ANSWERED, and `--api`
points that check at an arbitrary host — so an operator driving a REMOTE daemon from a box that
never ran `prsm setup` sails past it and dies on the dereference. Same class as the two sp1418
test-isolation bugs: code that assumes the developer's ambient ~/.prsm always exists.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from prsm.cli import main


def test_node_infer_without_an_identity_fails_cleanly_not_with_attributeerror():
    """No identity -> a clean, actionable error and a non-zero exit. Never a raw traceback.

    The identity guard fires AFTER the HuggingFace-deps check (the more fundamental precondition —
    no runtime, no inference), so this stubs `transformers` in sys.modules to deterministically
    clear that block and reach the guard. Stubbing (rather than importing the real transformers)
    matches the sibling sp644 tests and avoids their lazy-loader flakiness.
    """
    runner = CliRunner()

    peers = {"connected": [{"peer_id": "b" * 32, "address": "1.2.3.4:9001"}]}

    class _Resp:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return peers

    transformers_stub = MagicMock(
        AutoTokenizer=MagicMock(),
        AutoModelForCausalLM=MagicMock(),
    )

    # Daemon reachable (as --api at a remote host would be) + HF deps present, but NO local
    # identity. node_infer_cli imports load_node_identity function-locally, so patch it at the
    # SOURCE module (prsm.node.identity) — patching prsm.cli would not intercept the local import.
    with patch("prsm.node.identity.load_node_identity", return_value=None), \
         patch("httpx.get", return_value=_Resp()), \
         patch.dict("sys.modules", {"transformers": transformers_stub}):
        result = runner.invoke(
            main, ["node", "infer", "--prompt", "hi", "--api", "http://remote:8000"],
        )

    assert not isinstance(result.exception, AttributeError), (
        f"raw AttributeError leaked to the user instead of a clean error: {result.exception!r}"
    )
    assert result.exit_code != 0, "should fail — we cannot sign a receipt with no identity"

    combined = (result.stdout or "") + (result.output or "")
    assert "identity" in combined.lower(), (
        f"error does not mention the missing identity; user cannot act on it: {combined!r}"
    )
    assert "prsm setup" in combined.lower(), (
        f"error does not tell the user how to fix it: {combined!r}"
    )
