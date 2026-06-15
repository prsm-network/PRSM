"""
PRSM Cryptographic Key Management
================================

Production-grade key management system with secure key generation,
storage, rotation, and lifecycle management for enterprise deployment.
"""

import hashlib
import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, ed25519
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.fernet import Fernet

from prsm.core.database import db_manager
from prsm.core.config import get_settings
from .crypto_models import (
    KeyType, KeyUsage, CurveType, CryptoKey, CryptoKeyStore, KeyGenerationRequest
)

logger = structlog.get_logger(__name__)


class SecureKeyStorage:
    """Secure key storage with encryption at rest"""
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.cipher = Fernet(master_key)
        
    def encrypt_key_material(self, key_data: bytes) -> bytes:
        """Encrypt key material for storage"""
        return self.cipher.encrypt(key_data)
    
    def decrypt_key_material(self, encrypted_data: bytes) -> bytes:
        """Decrypt key material from storage"""
        return self.cipher.decrypt(encrypted_data)
    
    def generate_key_hash(self, key_data: bytes) -> str:
        """Sprint 1120 (Domain-08 review MEDIUM-6) — KEYED integrity tag over key
        material, HMAC-SHA256 under the master key (was a bare unkeyed sha256). An unkeyed
        hash is forgeable: an attacker with DB write could substitute key material AND
        recompute its sha256, passing the integrity check — making the ``is_compromised``
        signal meaningless. Keying it under the master key means a tamperer who doesn't
        hold the master key cannot forge a matching tag."""
        import hmac as _hmac
        return _hmac.new(self.master_key, key_data, hashlib.sha256).hexdigest()

    def verify_key_hash(self, key_data: bytes, expected_hex: str) -> bool:
        """Constant-time verification of the keyed integrity tag (closes the timing
        side-channel of a plain ``==`` on the hex digests)."""
        import hmac as _hmac
        return _hmac.compare_digest(
            self.generate_key_hash(key_data), str(expected_hex or ""),
        )

    def pkcs8_passphrase(self) -> bytes:
        """Sprint 1121 (Domain-08 review HIGH-4 remainder) — a passphrase for encrypting
        stored private keys at the PKCS8 layer, DERIVED from the master key with a
        DISTINCT domain label so it is NOT the same secret as the Fernet wrapper. This
        gives real defense-in-depth: an attacker who defeats the Fernet layer still faces
        a password-encrypted PKCS8 blob (whereas NoEncryption left plaintext key bytes)."""
        import hmac as _hmac
        return _hmac.new(
            self.master_key, b"prsm-pkcs8-key-encryption-v1", hashlib.sha256,
        ).digest()

    def reveal_private_pem(self, pem_bytes: bytes) -> bytes:
        """Transparently return the PLAINTEXT PKCS8 PEM for a stored private key, so
        callers of get_key see the same plaintext PEM as before. Encrypted PKCS8 (new,
        ``BEGIN ENCRYPTED PRIVATE KEY``) is decrypted with pkcs8_passphrase(); legacy
        NoEncryption PEM and symmetric raw bytes pass through unchanged (back-compat)."""
        try:
            head = bytes(pem_bytes)[:64]
        except Exception:  # noqa: BLE001
            return pem_bytes
        if b"ENCRYPTED PRIVATE KEY" not in head:
            return pem_bytes  # legacy plaintext PEM or symmetric raw bytes
        key = serialization.load_pem_private_key(
            bytes(pem_bytes), password=self.pkcs8_passphrase(),
        )
        return key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )


class KeyGenerator:
    """Cryptographic key generation utilities"""
    
    @staticmethod
    def generate_rsa_key(key_size: int = 2048) -> rsa.RSAPrivateKey:
        """Generate RSA key pair"""
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
    
    @staticmethod
    def generate_ecdsa_key(curve_type: CurveType = CurveType.SECP256R1) -> ec.EllipticCurvePrivateKey:
        """Generate ECDSA key pair"""
        curve_mapping = {
            CurveType.SECP256R1: ec.SECP256R1(),
            CurveType.SECP256K1: ec.SECP256K1(),
        }
        
        curve = curve_mapping.get(curve_type, ec.SECP256R1())
        return ec.generate_private_key(curve)
    
    @staticmethod
    def generate_ed25519_key() -> ed25519.Ed25519PrivateKey:
        """Generate Ed25519 key pair"""
        return ed25519.Ed25519PrivateKey.generate()
    
    @staticmethod
    def generate_symmetric_key(key_size: int = 32) -> bytes:
        """Generate symmetric encryption key"""
        return secrets.token_bytes(key_size)
    
    @staticmethod
    def derive_key_from_password(
        password: str, 
        salt: bytes, 
        key_length: int = 32,
        iterations: int = 100000
    ) -> bytes:
        """Derive key from password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=iterations
        )
        return kdf.derive(password.encode())
    
    @staticmethod
    def derive_key_from_secret(
        secret: bytes,
        salt: bytes,
        key_length: int = 32,
        n: int = 2**14,
        r: int = 8,
        p: int = 1
    ) -> bytes:
        """Derive key using Scrypt (memory-hard function)"""
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            n=n,
            r=r,
            p=p
        )
        return kdf.derive(secret)


class KeyManager:
    """
    Production-grade cryptographic key management system
    
    🔐 KEY MANAGEMENT FEATURES:
    - Secure key generation and storage
    - Hardware-backed key support
    - Automated key rotation
    - Key derivation and hierarchical deterministic keys
    - Enterprise-grade access controls
    - Comprehensive audit logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        
        # Initialize master key for encryption at rest
        self._initialize_master_key()
        self.storage = SecureKeyStorage(self.master_key)
        self.generator = KeyGenerator()
        
        # Key rotation settings
        self.default_key_lifetime = timedelta(days=self.config.get("default_key_lifetime_days", 365))
        self.rotation_check_interval = timedelta(hours=self.config.get("rotation_check_hours", 24))
        
        # Security settings
        self.require_hardware_backing = self.config.get("require_hardware_backing", False)
        self.enable_key_escrow = self.config.get("enable_key_escrow", False)
        self.max_key_derivations = self.config.get("max_key_derivations", 1000)
        
        print("🔐 Key Manager initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default key management configuration"""
        settings = get_settings()
        
        return {
            "master_key_source": "environment",  # environment, hsm, kms
            "default_key_lifetime_days": 365,
            "rotation_check_hours": 24,
            "require_hardware_backing": False,
            "enable_key_escrow": False,
            "max_key_derivations": 1000,
            "key_backup_enabled": True,
            "compliance_mode": "standard"  # standard, fips, common_criteria
        }
    
    def _initialize_master_key(self):
        """Initialize master key for key encryption"""
        master_key_source = self.config.get("master_key_source", "environment")
        
        if master_key_source == "environment":
            # Get master key from environment variable
            master_key_b64 = os.getenv("PRSM_MASTER_KEY")
            if master_key_b64:
                import base64
                self.master_key = base64.b64decode(master_key_b64)
            else:
                # sp1122 (Domain-08 review HIGH-4 remainder) — in PRODUCTION a missing
                # PRSM_MASTER_KEY must FAIL CLOSED, not silently mint a per-process-random
                # key. The random fallback makes every previously-stored encrypted key
                # permanently undecryptable after a restart (silent data loss) and gives
                # no real at-rest protection. Mirrors the schemas.py jwt-secret
                # production gate (PRSM_ENV). Dev/test (the default env) keep the
                # convenient ephemeral key + warning.
                if os.getenv("PRSM_ENV", "development").lower() == "production":
                    raise RuntimeError(
                        "FATAL: PRSM_MASTER_KEY is required in production "
                        "(PRSM_ENV=production) — refusing to start with a per-process "
                        "random key that would make stored keys undecryptable after a "
                        "restart. Set PRSM_MASTER_KEY to a stable Fernet key "
                        "(Fernet.generate_key())."
                    )
                # Generate new master key (for development)
                self.master_key = Fernet.generate_key()
                logger.warning("Generated new master key - store securely in production")
        else:
            # sp1106 (Domain-08 review HIGH-4) — selecting an HSM/KMS master-key source
            # asserts HARDWARE-backed key protection. There is no real HSM/KMS
            # integration yet, and silently minting an ephemeral in-process Fernet key
            # (the prior behavior) gave a FALSE sense of hardware protection while the
            # key was software-only and lost on restart — a fail-OPEN on a
            # security-relevant config. Fail CLOSED: refuse to start under a security
            # posture we can't actually provide.
            raise NotImplementedError(
                f"master_key_source={master_key_source!r} requests hardware-backed key "
                "protection (HSM/KMS), but no HSM/KMS integration is implemented. "
                "Refusing to silently fall back to a software key that would falsely "
                "advertise hardware protection. Use master_key_source='environment' "
                "with a securely-provisioned PRSM_MASTER_KEY, or implement the HSM/KMS "
                "backend."
            )
    
    async def generate_key(self, request: KeyGenerationRequest) -> CryptoKey:
        """Generate a new cryptographic key"""
        try:
            # Generate the actual key material
            key_material = await self._generate_key_material(request)
            
            # Serialize key for storage
            private_key_bytes, public_key_bytes = await self._serialize_key_material(
                key_material, request.key_type
            )
            
            # Encrypt key material
            encrypted_private_key = self.storage.encrypt_key_material(private_key_bytes) if private_key_bytes else None
            encrypted_public_key = self.storage.encrypt_key_material(public_key_bytes) if public_key_bytes else None
            
            # Generate key hash for integrity
            key_hash_source = private_key_bytes or public_key_bytes
            key_material_hash = self.storage.generate_key_hash(key_hash_source)
            
            # Calculate expiration
            expires_at = None
            if request.expires_in_days:
                expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)
            else:
                expires_at = datetime.now(timezone.utc) + self.default_key_lifetime
            
            # Store in database
            async with db_manager.session() as session:
                db_key = CryptoKeyStore(
                    key_name=request.key_name,
                    user_id=request.user_id,
                    key_type=request.key_type.value,
                    algorithm=request.algorithm,
                    key_usage=request.key_usage.value,
                    curve_type=request.curve_type.value if request.curve_type else None,
                    key_size=request.key_size,
                    is_hardware_backed=request.is_hardware_backed,
                    is_exportable=request.is_exportable,
                    encrypted_private_key=encrypted_private_key,
                    encrypted_public_key=encrypted_public_key,
                    key_material_hash=key_material_hash,
                    expires_at=expires_at,
                    metadata=request.metadata,
                    tags=request.tags
                )
                
                session.add(db_key)
                session.commit()
                session.refresh(db_key)
                
                logger.info("Cryptographic key generated",
                          key_id=str(db_key.key_id),
                          key_name=request.key_name,
                          key_type=request.key_type.value,
                          user_id=request.user_id)
                
                return CryptoKey(
                    key_id=str(db_key.key_id),
                    key_name=db_key.key_name,
                    key_type=KeyType(db_key.key_type),
                    algorithm=db_key.algorithm,
                    key_usage=KeyUsage(db_key.key_usage),
                    curve_type=CurveType(db_key.curve_type) if db_key.curve_type else None,
                    key_size=db_key.key_size,
                    is_hardware_backed=db_key.is_hardware_backed,
                    is_exportable=db_key.is_exportable,
                    created_at=db_key.created_at,
                    expires_at=db_key.expires_at,
                    is_active=db_key.is_active,
                    metadata=db_key.metadata
                )
                
        except Exception as e:
            logger.error("Key generation failed", error=str(e))
            raise
    
    async def get_key(self, key_id: str) -> Optional[CryptoKey]:
        """Retrieve key information"""
        try:
            async with db_manager.session() as session:
                db_key = session.query(CryptoKeyStore).filter(
                    CryptoKeyStore.key_id == key_id,
                    CryptoKeyStore.is_active == True
                ).first()
                
                if not db_key:
                    return None
                
                # Update last used timestamp
                db_key.last_used_at = datetime.now(timezone.utc)
                session.commit()
                
                return CryptoKey(
                    key_id=str(db_key.key_id),
                    key_name=db_key.key_name,
                    key_type=KeyType(db_key.key_type),
                    algorithm=db_key.algorithm,
                    key_usage=KeyUsage(db_key.key_usage),
                    curve_type=CurveType(db_key.curve_type) if db_key.curve_type else None,
                    key_size=db_key.key_size,
                    is_hardware_backed=db_key.is_hardware_backed,
                    is_exportable=db_key.is_exportable,
                    created_at=db_key.created_at,
                    expires_at=db_key.expires_at,
                    is_active=db_key.is_active,
                    metadata=db_key.metadata
                )
                
        except Exception as e:
            logger.error("Failed to retrieve key", key_id=key_id, error=str(e))
            return None
    
    async def get_key_material(self, key_id: str, usage_context: str) -> Optional[bytes]:
        """Retrieve decrypted key material for use (audit logged)"""
        try:
            async with db_manager.session() as session:
                db_key = session.query(CryptoKeyStore).filter(
                    CryptoKeyStore.key_id == key_id,
                    CryptoKeyStore.is_active == True
                ).first()
                
                if not db_key:
                    logger.warning("Key not found for material retrieval", key_id=key_id)
                    return None
                
                # Check expiration
                if db_key.expires_at and datetime.now(timezone.utc) > db_key.expires_at:
                    logger.warning("Attempted to use expired key", key_id=key_id)
                    return None
                
                # Decrypt key material
                if db_key.encrypted_private_key:
                    key_material = self.storage.decrypt_key_material(db_key.encrypted_private_key)
                elif db_key.encrypted_public_key:
                    key_material = self.storage.decrypt_key_material(db_key.encrypted_public_key)
                else:
                    logger.error("No key material found", key_id=key_id)
                    return None
                
                # Verify integrity (sp1120 — keyed HMAC + constant-time compare).
                if not self.storage.verify_key_hash(
                    key_material, db_key.key_material_hash,
                ):
                    logger.error("Key material integrity check failed", key_id=key_id)
                    # Mark key as compromised
                    db_key.is_compromised = True
                    session.commit()
                    return None
                
                # sp1121 — transparently decrypt the PKCS8 layer (if present) so callers
                # receive the plaintext PEM exactly as before. Integrity was already
                # verified above over the stored (encrypted) form.
                key_material = self.storage.reveal_private_pem(key_material)

                # Update usage tracking
                db_key.last_used_at = datetime.now(timezone.utc)
                session.commit()

                # Audit log key usage
                logger.info("Key material accessed",
                          key_id=key_id,
                          key_name=db_key.key_name,
                          usage_context=usage_context,
                          user_id=db_key.user_id)
                
                return key_material
                
        except Exception as e:
            logger.error("Failed to retrieve key material", key_id=key_id, error=str(e))
            return None
    
    async def rotate_key(self, key_id: str) -> Optional[CryptoKey]:
        """Rotate an existing key"""
        try:
            # Get current key
            current_key = await self.get_key(key_id)
            if not current_key:
                return None
            
            # Create rotation request based on current key
            rotation_request = KeyGenerationRequest(
                key_name=f"{current_key.key_name}_rotated_{datetime.now().strftime('%Y%m%d')}",
                key_type=current_key.key_type,
                algorithm=current_key.algorithm,
                key_usage=current_key.key_usage,
                curve_type=current_key.curve_type,
                key_size=current_key.key_size,
                user_id=current_key.metadata.get("user_id"),
                is_hardware_backed=current_key.is_hardware_backed,
                is_exportable=current_key.is_exportable,
                metadata={**current_key.metadata, "rotated_from": key_id}
            )
            
            # Generate new key
            new_key = await self.generate_key(rotation_request)
            
            # Deactivate old key
            await self._deactivate_key(key_id, "rotated")
            
            logger.info("Key rotated successfully",
                      old_key_id=key_id,
                      new_key_id=new_key.key_id)
            
            return new_key
            
        except Exception as e:
            logger.error("Key rotation failed", key_id=key_id, error=str(e))
            return None
    
    async def list_keys(
        self, 
        user_id: Optional[str] = None,
        key_usage: Optional[KeyUsage] = None,
        include_inactive: bool = False
    ) -> List[CryptoKey]:
        """List keys with filtering"""
        try:
            async with db_manager.session() as session:
                query = session.query(CryptoKeyStore)
                
                if user_id:
                    query = query.filter(CryptoKeyStore.user_id == user_id)
                if key_usage:
                    query = query.filter(CryptoKeyStore.key_usage == key_usage.value)
                if not include_inactive:
                    query = query.filter(CryptoKeyStore.is_active == True)
                
                db_keys = query.order_by(CryptoKeyStore.created_at.desc()).all()
                
                return [
                    CryptoKey(
                        key_id=str(key.key_id),
                        key_name=key.key_name,
                        key_type=KeyType(key.key_type),
                        algorithm=key.algorithm,
                        key_usage=KeyUsage(key.key_usage),
                        curve_type=CurveType(key.curve_type) if key.curve_type else None,
                        key_size=key.key_size,
                        is_hardware_backed=key.is_hardware_backed,
                        is_exportable=key.is_exportable,
                        created_at=key.created_at,
                        expires_at=key.expires_at,
                        is_active=key.is_active,
                        metadata=key.metadata
                    )
                    for key in db_keys
                ]
                
        except Exception as e:
            logger.error("Failed to list keys", error=str(e))
            return []
    
    async def check_key_rotation_due(self) -> List[str]:
        """Check for keys that need rotation"""
        try:
            async with db_manager.session() as session:
                # Keys expiring in the next 30 days
                rotation_threshold = datetime.now(timezone.utc) + timedelta(days=30)
                
                keys_due = session.query(CryptoKeyStore).filter(
                    CryptoKeyStore.is_active == True,
                    CryptoKeyStore.expires_at <= rotation_threshold
                ).all()
                
                return [str(key.key_id) for key in keys_due]
                
        except Exception as e:
            logger.error("Failed to check key rotation", error=str(e))
            return []
    
    async def derive_key(
        self,
        parent_key_id: str,
        derivation_path: str,
        child_key_name: str,
        key_usage: KeyUsage
    ) -> Optional[CryptoKey]:
        """Derive child key from parent key"""
        try:
            # Get parent key material
            parent_material = await self.get_key_material(parent_key_id, "key_derivation")
            if not parent_material:
                return None
            
            # Derive child key using HKDF or similar
            import hashlib
            child_material = hashlib.pbkdf2_hmac(
                'sha256',
                parent_material,
                derivation_path.encode(),
                100000,
                32
            )
            
            # Create derived key request
            derived_request = KeyGenerationRequest(
                key_name=child_key_name,
                key_type=KeyType.AES,
                algorithm="aes_256",
                key_usage=key_usage,
                key_size=256,
                metadata={
                    "parent_key_id": parent_key_id,
                    "derivation_path": derivation_path,
                    "derived": True
                }
            )
            
            # Store derived key (this would need modification to handle pre-generated material)
            # For now, generate a new key with the derived material in metadata
            return await self.generate_key(derived_request)
            
        except Exception as e:
            logger.error("Key derivation failed", 
                        parent_key_id=parent_key_id, 
                        derivation_path=derivation_path, 
                        error=str(e))
            return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get key management system status"""
        try:
            async with db_manager.session() as session:
                total_keys = session.query(CryptoKeyStore).count()
                active_keys = session.query(CryptoKeyStore).filter(
                    CryptoKeyStore.is_active == True
                ).count()
                hardware_backed = session.query(CryptoKeyStore).filter(
                    CryptoKeyStore.is_hardware_backed == True,
                    CryptoKeyStore.is_active == True
                ).count()
                compromised_keys = session.query(CryptoKeyStore).filter(
                    CryptoKeyStore.is_compromised == True
                ).count()
                
                # Keys expiring soon
                expiring_soon = session.query(CryptoKeyStore).filter(
                    CryptoKeyStore.is_active == True,
                    CryptoKeyStore.expires_at <= datetime.now(timezone.utc) + timedelta(days=30)
                ).count()
                
                return {
                    "system_health": "healthy" if compromised_keys == 0 else "warning",
                    "total_keys": total_keys,
                    "active_keys": active_keys,
                    "hardware_backed_keys": hardware_backed,
                    "compromised_keys": compromised_keys,
                    "keys_expiring_soon": expiring_soon,
                    "master_key_status": "active",
                    "compliance_mode": self.config.get("compliance_mode", "standard"),
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            return {
                "system_health": "error",
                "error": str(e),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
    
    # Internal helper methods
    
    async def _generate_key_material(self, request: KeyGenerationRequest) -> Any:
        """Generate actual cryptographic key material"""
        if request.key_type == KeyType.RSA:
            return self.generator.generate_rsa_key(request.key_size or 2048)
        elif request.key_type == KeyType.ECDSA:
            return self.generator.generate_ecdsa_key(request.curve_type or CurveType.SECP256R1)
        elif request.key_type == KeyType.ED25519:
            return self.generator.generate_ed25519_key()
        elif request.key_type == KeyType.AES:
            return self.generator.generate_symmetric_key(request.key_size // 8 if request.key_size else 32)
        elif request.key_type == KeyType.CHACHA20:
            return self.generator.generate_symmetric_key(32)
        else:
            raise ValueError(f"Unsupported key type: {request.key_type}")
    
    async def _serialize_key_material(self, key_material: Any, key_type: KeyType) -> tuple:
        """Serialize key material for storage"""
        if key_type in [KeyType.RSA, KeyType.ECDSA, KeyType.ED25519]:
            # Asymmetric keys
            # sp1121 (Domain-08 review HIGH-4 remainder) — encrypt the PKCS8 at rest with
            # a master-key-DERIVED passphrase (BestAvailableEncryption) instead of
            # NoEncryption. The Fernet wrapper still applies on top; this is the
            # defense-in-depth layer so a Fernet-layer compromise doesn't expose plaintext
            # private keys. get_key transparently reveals the plaintext PEM to callers.
            private_key_bytes = key_material.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.BestAvailableEncryption(
                    self.storage.pkcs8_passphrase(),
                ),
            )
            public_key_bytes = key_material.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return private_key_bytes, public_key_bytes
        else:
            # Symmetric keys
            return key_material, None
    
    async def _deactivate_key(self, key_id: str, reason: str):
        """Deactivate a key"""
        async with db_manager.session() as session:
            db_key = session.query(CryptoKeyStore).filter(
                CryptoKeyStore.key_id == key_id
            ).first()
            
            if db_key:
                db_key.is_active = False
                db_key.metadata = {**db_key.metadata, "deactivation_reason": reason, "deactivated_at": datetime.now(timezone.utc).isoformat()}
                session.commit()


# Global key manager instance
_key_manager: Optional[KeyManager] = None

async def get_key_manager() -> KeyManager:
    """Get or create the global key manager instance"""
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager()
    return _key_manager