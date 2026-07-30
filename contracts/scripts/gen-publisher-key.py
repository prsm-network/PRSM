#!/usr/bin/env python3
"""Sprint 1481 — generate the OperatorRewardPool epoch-job publisher key.

    python3 contracts/scripts/gen-publisher-key.py

Run this ON THE HOST THAT WILL RUN THE EPOCH JOB. It generates a fresh EOA, writes
the private key to ~/.prsm/epoch_publisher.key at mode 600, and prints ONLY the
address. The key never passes through argv, so it cannot land in shell history or
a process listing.

Why a dedicated key rather than reusing an existing one: the publisher is an
always-on hot key on a server. `OperatorRewardPool` deliberately separates owner
from publisher so the publisher can set epoch roots but CANNOT move funds
(sweepSurplus and setRootPublisher both reject it — covered by the contract tests).
Reusing the mainnet deployer or a balance-holding key would defeat that split: a
compromised epoch host would then also surrender the identity that deploys your
production contracts.

Blast radius if this key leaks: bad epoch roots get published — which the Safe can
correct by calling setRootPublisher (owner-only). Funds are not reachable.

Refuses to overwrite an existing key: once this address is set as rootPublisher and
funded, clobbering it would leave a funded address whose key you no longer hold
while the contract still points at it.
"""
import os
import stat
import sys
from pathlib import Path

KEY_PATH = Path.home() / ".prsm" / "epoch_publisher.key"


def main() -> int:
    try:
        from eth_account import Account
    except ImportError:
        print("ERROR: eth_account not installed. Try: pip install eth-account")
        return 2

    if KEY_PATH.exists():
        print(f"REFUSING: {KEY_PATH} already exists — not overwriting.")
        print("If you truly want a NEW publisher key, move the old one aside first")
        print("and remember to re-point the contract via Safe.setRootPublisher().")
        return 1

    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)                       # nothing group/world-readable, even transiently
    acct = Account.create()
    # Write with restrictive perms from the start rather than chmod-ing after, so the
    # key is never briefly readable by another user on a shared host.
    fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("0x" + acct.key.hex())
    KEY_PATH.chmod(0o600)

    mode = stat.filemode(KEY_PATH.stat().st_mode)
    print(f"PUBLISHER ADDRESS: {acct.address}")
    print(f"key written to   : {KEY_PATH}  ({mode})")
    print()
    print("NEXT:")
    print("  1. Share the ADDRESS above (public). Keep the file secret.")
    print("  2. Fund the address with a small amount of Base ETH (one cheap tx/epoch).")
    print(f"  3. Deploy with REWARD_POOL_PUBLISHER={acct.address}")
    print("  4. The epoch job reads the key FROM THE FILE — never paste it anywhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
