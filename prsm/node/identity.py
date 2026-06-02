"""
Node Identity
=============

Ed25519 keypair generation, persistence, and message signing.
Each PRSM node has a unique identity derived from its public key.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
import base64


@dataclass
class NodeIdentity:
    """Represents a node's cryptographic identity on the network."""
    node_id: str                             # hex(sha256(public_key_bytes))[:32]
    public_key_bytes: bytes = field(repr=False)
    private_key_bytes: bytes = field(repr=False)
    display_name: str = "prsm-node"
    created_at: float = field(default_factory=time.time)
    # Phase 3.1: optional Ethereum address for on-chain settlement
    # participation. Separate from the Ed25519 node identity — set by
    # operators who want to participate in batched on-chain settlement
    # (Phase 3.1) or on-chain governance. None for Phase 2/3 nodes.
    ethereum_address: Optional[str] = None

    # Cached key objects (not serialized)
    _private_key: Optional[Ed25519PrivateKey] = field(default=None, repr=False)
    _public_key: Optional[Ed25519PublicKey] = field(default=None, repr=False)

    @property
    def private_key(self) -> Ed25519PrivateKey:
        if self._private_key is None:
            self._private_key = Ed25519PrivateKey.from_private_bytes(self.private_key_bytes)
        return self._private_key

    @property
    def public_key(self) -> Ed25519PublicKey:
        if self._public_key is None:
            self._public_key = self.private_key.public_key()
        return self._public_key

    @property
    def public_key_b64(self) -> str:
        """Base64-encoded public key for sharing."""
        return base64.b64encode(self.public_key_bytes).decode()

    def sign(self, data: bytes) -> str:
        """Sign data with this node's private key, return base64 signature."""
        signature = self.private_key.sign(data)
        return base64.b64encode(signature).decode()

    def verify(self, data: bytes, signature_b64: str) -> bool:
        """Verify a signature against this node's public key."""
        try:
            sig_bytes = base64.b64decode(signature_b64)
            self.public_key.verify(sig_bytes, data)
            return True
        except Exception:
            return False

    def to_dict(self) -> dict:
        """Serialize identity for persistence (includes private key)."""
        return {
            "node_id": self.node_id,
            "public_key": base64.b64encode(self.public_key_bytes).decode(),
            "private_key": base64.b64encode(self.private_key_bytes).decode(),
            "display_name": self.display_name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NodeIdentity":
        """Deserialize identity from persisted data."""
        return cls(
            node_id=data["node_id"],
            public_key_bytes=base64.b64decode(data["public_key"]),
            private_key_bytes=base64.b64decode(data["private_key"]),
            display_name=data.get("display_name", "prsm-node"),
            created_at=data.get("created_at", time.time()),
        )


def generate_node_identity(display_name: str = "prsm-node") -> NodeIdentity:
    """Generate a new Ed25519 keypair and derive node_id from public key hash."""
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    node_id = hashlib.sha256(public_bytes).hexdigest()[:32]

    return NodeIdentity(
        node_id=node_id,
        public_key_bytes=public_bytes,
        private_key_bytes=private_bytes,
        display_name=display_name,
        created_at=time.time(),
    )


def save_node_identity(identity: NodeIdentity, path: Path) -> None:
    """Save identity to a JSON file with OWNER-ONLY permissions.

    sp942 — the identity JSON contains the node's ed25519 PRIVATE key, so it is
    written 0o600 (and its parent dir 0o700). The previous ``path.write_text``
    left the key world-readable (default 0o644), exposing the node's signing key
    to any other local user / co-tenant container process. The file is created
    with 0o600 BEFORE the key bytes are written (no world-readable window for a
    new file); a chmod afterwards also tightens a pre-existing file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:  # pragma: no cover — non-POSIX / permission-denied; best-effort
        pass
    payload = json.dumps(identity.to_dict(), indent=2)
    # os.open with mode 0o600 sets perms atomically on a NEW file (no 0o644
    # window); for a pre-existing file the mode arg is ignored, so chmod after.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
    finally:
        try:
            path.chmod(0o600)
        except OSError:  # pragma: no cover
            pass


def load_node_identity(path: Path) -> Optional[NodeIdentity]:
    """Load identity from a JSON file, return None if not found."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return NodeIdentity.from_dict(data)


def verify_signature(public_key_b64: str, data: bytes, signature_b64: str) -> bool:
    """Verify a signature using a base64-encoded public key (for verifying other nodes)."""
    try:
        pub_bytes = base64.b64decode(public_key_b64)
        pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
        sig_bytes = base64.b64decode(signature_b64)
        pub_key.verify(sig_bytes, data)
        return True
    except Exception:
        return False
