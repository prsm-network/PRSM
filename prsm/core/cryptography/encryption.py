"""
PRSM Encryption Service
======================

Production-grade encryption service providing symmetric and asymmetric
encryption, secure data storage, and comprehensive cryptographic operations.
"""

import hashlib
import json
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

import structlog
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.fernet import Fernet

from prsm.core.database import db_manager
from .key_management import get_key_manager
from .crypto_models import (
    EncryptionAlgorithm, PrivacyLevel, EncryptionResult, DecryptionResult, EncryptedData,
    EncryptionRequest, DecryptionRequest
)

logger = structlog.get_logger(__name__)


class CipherEngine:
    """Core encryption/decryption engine"""
    
    @staticmethod
    def encrypt_aes_gcm(data: bytes, key: bytes, associated_data: Optional[bytes] = None) -> Dict[str, bytes]:
        """Encrypt data using AES-GCM"""
        cipher = AESGCM(key)
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        ciphertext = cipher.encrypt(nonce, data, associated_data)
        
        return {
            "ciphertext": ciphertext,
            "nonce": nonce,
            "tag": b""  # Tag is included in ciphertext for AESGCM
        }
    
    @staticmethod
    def decrypt_aes_gcm(ciphertext: bytes, key: bytes, nonce: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using AES-GCM"""
        cipher = AESGCM(key)
        return cipher.decrypt(nonce, ciphertext, associated_data)
    
    @staticmethod
    def encrypt_chacha20_poly1305(data: bytes, key: bytes, associated_data: Optional[bytes] = None) -> Dict[str, bytes]:
        """Encrypt data using ChaCha20-Poly1305"""
        cipher = ChaCha20Poly1305(key)
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        
        ciphertext = cipher.encrypt(nonce, data, associated_data)
        
        return {
            "ciphertext": ciphertext,
            "nonce": nonce,
            "tag": b""  # Tag is included in ciphertext
        }
    
    @staticmethod
    def decrypt_chacha20_poly1305(ciphertext: bytes, key: bytes, nonce: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using ChaCha20-Poly1305"""
        cipher = ChaCha20Poly1305(key)
        return cipher.decrypt(nonce, ciphertext, associated_data)
    
    @staticmethod
    def encrypt_rsa_oaep(data: bytes, public_key: rsa.RSAPublicKey) -> bytes:
        """Encrypt data using RSA-OAEP"""
        return public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    @staticmethod
    def decrypt_rsa_oaep(ciphertext: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """Decrypt data using RSA-OAEP"""
        return private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    @staticmethod
    def encrypt_fernet(data: bytes, key: bytes) -> bytes:
        """Encrypt data using Fernet (includes MAC)"""
        f = Fernet(key)
        return f.encrypt(data)
    
    @staticmethod
    def decrypt_fernet(ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt data using Fernet"""
        f = Fernet(key)
        return f.decrypt(ciphertext)


class EncryptionService:
    """
    Production-grade encryption service
    
    🔒 ENCRYPTION FEATURES:
    - Multi-algorithm symmetric and asymmetric encryption
    - Secure key derivation and management integration
    - Data-at-rest encryption with integrity protection
    - Privacy-level based encryption policies
    - Comprehensive audit logging and compliance
    - Performance-optimized cryptographic operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.cipher_engine = CipherEngine()
        self.key_manager = None
        
        # Algorithm preferences by privacy level
        self.algorithm_preferences = {
            PrivacyLevel.PUBLIC: EncryptionAlgorithm.FERNET,
            PrivacyLevel.CONFIDENTIAL: EncryptionAlgorithm.AES_256_GCM,
            PrivacyLevel.SECRET: EncryptionAlgorithm.CHACHA20_POLY1305,
            PrivacyLevel.TOP_SECRET: EncryptionAlgorithm.AES_256_GCM,
            PrivacyLevel.ZERO_KNOWLEDGE: EncryptionAlgorithm.XCHACHA20_POLY1305
        }
        
        # Performance settings
        self.chunk_size = self.config.get("chunk_size", 64 * 1024)  # 64KB chunks
        self.enable_compression = self.config.get("enable_compression", True)
        
        print("🔒 Encryption Service initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default encryption configuration"""
        return {
            "default_algorithm": "aes_256_gcm",
            "chunk_size": 64 * 1024,
            "enable_compression": True,
            "integrity_check_enabled": True,
            "performance_mode": "balanced",  # fast, balanced, secure
            "compliance_mode": "standard"
        }
    
    async def initialize(self):
        """Initialize encryption service"""
        self.key_manager = await get_key_manager()
        logger.info("✅ Encryption service initialized")
    
    async def encrypt(self, request: EncryptionRequest) -> EncryptionResult:
        """Encrypt data with specified algorithm and key"""
        start_time = datetime.now()
        
        try:
            # Get encryption key
            key_material = await self.key_manager.get_key_material(request.key_id, "encryption")
            if not key_material:
                return EncryptionResult(
                    success=False,
                    error_message="Encryption key not found or inaccessible"
                )
            
            # Prepare data for encryption
            data_bytes = request.data.encode('utf-8') if isinstance(request.data, str) else request.data
            
            # Optional compression
            if self.enable_compression and len(data_bytes) > 1024:
                import gzip
                data_bytes = gzip.compress(data_bytes)
                compression_used = True
            else:
                compression_used = False
            
            # Generate content hash for integrity
            content_hash = hashlib.sha256(data_bytes).hexdigest()
            
            # Encrypt based on algorithm
            encryption_result = await self._encrypt_with_algorithm(
                data_bytes, 
                key_material, 
                request.algorithm
            )
            
            if not encryption_result["success"]:
                return EncryptionResult(
                    success=False,
                    error_message=encryption_result["error"]
                )
            
            # Store encrypted data in database
            data_id = await self._store_encrypted_data(
                request, 
                encryption_result, 
                content_hash, 
                compression_used
            )
            
            operation_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info("Data encrypted successfully",
                      data_id=data_id,
                      algorithm=request.algorithm.value,
                      data_size=len(data_bytes),
                      operation_time_ms=operation_time)
            
            return EncryptionResult(
                success=True,
                encrypted_data=data_id,  # Return data ID, not raw encrypted data
                algorithm=request.algorithm,
                initialization_vector=encryption_result.get("iv_b64"),
                authentication_tag=encryption_result.get("tag_b64"),
                key_id=request.key_id,
                content_hash=content_hash,
                encryption_context={
                    "privacy_level": request.privacy_level.value,
                    "content_type": request.content_type,
                    "compression_used": compression_used,
                    **request.encryption_context
                },
                operation_time_ms=operation_time
            )
            
        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            return EncryptionResult(
                success=False,
                error_message=f"Encryption error: {str(e)}"
            )
    
    async def decrypt(self, request: DecryptionRequest) -> DecryptionResult:
        """Decrypt data by encrypted data ID"""
        start_time = datetime.now()
        
        try:
            # Retrieve encrypted data from database
            encrypted_record = await self._get_encrypted_data(request.encrypted_data_id)
            if not encrypted_record:
                return DecryptionResult(
                    success=False,
                    error_message="Encrypted data not found"
                )
            
            # Get decryption key
            key_material = await self.key_manager.get_key_material(
                str(encrypted_record.key_id), 
                "decryption"
            )
            if not key_material:
                return DecryptionResult(
                    success=False,
                    error_message="Decryption key not found or inaccessible"
                )
            
            # Decrypt based on algorithm
            algorithm = EncryptionAlgorithm(encrypted_record.algorithm)
            decryption_result = await self._decrypt_with_algorithm(
                encrypted_record, 
                key_material, 
                algorithm
            )
            
            if not decryption_result["success"]:
                return DecryptionResult(
                    success=False,
                    error_message=decryption_result["error"]
                )
            
            decrypted_bytes = decryption_result["data"]
            
            # Handle decompression if used
            compression_used = encrypted_record.encryption_context.get("compression_used", False)
            if compression_used:
                import gzip
                decrypted_bytes = gzip.decompress(decrypted_bytes)
            
            # Verify content integrity. sp1120 (Domain-08 review MEDIUM-6) — constant-time
            # compare of the digests (closes the timing side-channel of a plain ==). The
            # AEAD tag on the ciphertext is the primary integrity protection; this
            # checksum is defense-in-depth, compared in constant time here.
            import hmac as _hmac
            computed_hash = hashlib.sha256(decrypted_bytes).hexdigest()
            content_verified = _hmac.compare_digest(
                computed_hash, str(encrypted_record.content_hash or ""),
            )
            
            if not content_verified:
                logger.error("Content integrity verification failed",
                           encrypted_data_id=request.encrypted_data_id)
                return DecryptionResult(
                    success=False,
                    error_message="Content integrity verification failed"
                )
            
            # Convert back to string if it was originally text
            if encrypted_record.content_type.startswith("text/"):
                decrypted_data = decrypted_bytes.decode('utf-8')
            else:
                # For binary data, return base64 encoded
                import base64
                decrypted_data = base64.b64encode(decrypted_bytes).decode('ascii')
            
            operation_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Update access timestamp
            await self._update_access_timestamp(request.encrypted_data_id)
            
            logger.info("Data decrypted successfully",
                      encrypted_data_id=request.encrypted_data_id,
                      algorithm=algorithm.value,
                      operation_time_ms=operation_time)
            
            return DecryptionResult(
                success=True,
                decrypted_data=decrypted_data,
                content_verified=content_verified,
                key_id=str(encrypted_record.key_id),
                decryption_context={
                    "privacy_level": encrypted_record.privacy_level,
                    "content_type": encrypted_record.content_type,
                    "compression_used": compression_used,
                    **request.decryption_context
                },
                operation_time_ms=operation_time
            )
            
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            return DecryptionResult(
                success=False,
                error_message=f"Decryption error: {str(e)}"
            )
    
    async def encrypt_large_data(
        self, 
        data: bytes, 
        key_id: str, 
        algorithm: EncryptionAlgorithm,
        chunk_callback: Optional[callable] = None
    ) -> EncryptionResult:
        """Encrypt large data using streaming encryption"""
        try:
            # For large data, use chunked encryption
            encrypted_chunks = []
            total_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size
            
            for i in range(0, len(data), self.chunk_size):
                chunk = data[i:i + self.chunk_size]
                chunk_request = EncryptionRequest(
                    data=chunk,
                    key_id=key_id,
                    algorithm=algorithm,
                    content_type="application/octet-stream",
                    encryption_context={"chunk_index": len(encrypted_chunks), "total_chunks": total_chunks}
                )
                
                chunk_result = await self.encrypt(chunk_request)
                if not chunk_result.success:
                    return chunk_result
                
                encrypted_chunks.append(chunk_result.encrypted_data)
                
                if chunk_callback:
                    chunk_callback(len(encrypted_chunks), total_chunks)
            
            # Create combined result
            return EncryptionResult(
                success=True,
                encrypted_data=json.dumps({"chunks": encrypted_chunks, "chunk_size": self.chunk_size}),
                algorithm=algorithm,
                key_id=key_id,
                encryption_context={"chunked_encryption": True, "total_chunks": total_chunks}
            )
            
        except Exception as e:
            logger.error("Large data encryption failed", error=str(e))
            return EncryptionResult(
                success=False,
                error_message=f"Large data encryption error: {str(e)}"
            )
    
    async def get_encryption_status(self, data_id: str) -> Dict[str, Any]:
        """Get encryption status and metadata"""
        try:
            encrypted_record = await self._get_encrypted_data(data_id)
            if not encrypted_record:
                return {"found": False}
            
            return {
                "found": True,
                "data_id": str(encrypted_record.data_id),
                "algorithm": encrypted_record.algorithm,
                "content_type": encrypted_record.content_type,
                "privacy_level": encrypted_record.privacy_level,
                "owner_id": encrypted_record.owner_id,
                "created_at": encrypted_record.created_at.isoformat(),
                "last_accessed": encrypted_record.accessed_at.isoformat() if encrypted_record.accessed_at else None,
                "expires_at": encrypted_record.expires_at.isoformat() if encrypted_record.expires_at else None,
                "encryption_context": encrypted_record.encryption_context,
                "metadata": encrypted_record.metadata
            }
            
        except Exception as e:
            logger.error("Failed to get encryption status", data_id=data_id, error=str(e))
            return {"found": False, "error": str(e)}
    
    async def list_encrypted_data(
        self, 
        owner_id: str, 
        privacy_level: Optional[PrivacyLevel] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List encrypted data for an owner"""
        try:
            async with db_manager.session() as session:
                query = session.query(EncryptedData).filter(
                    EncryptedData.owner_id == owner_id
                )
                
                if privacy_level:
                    query = query.filter(EncryptedData.privacy_level == privacy_level.value)
                
                records = query.order_by(EncryptedData.created_at.desc()).limit(limit).all()
                
                return [
                    {
                        "data_id": str(record.data_id),
                        "content_type": record.content_type,
                        "privacy_level": record.privacy_level,
                        "algorithm": record.algorithm,
                        "created_at": record.created_at.isoformat(),
                        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
                        "metadata": record.metadata
                    }
                    for record in records
                ]
                
        except Exception as e:
            logger.error("Failed to list encrypted data", owner_id=owner_id, error=str(e))
            return []
    
    # Internal helper methods
    
    async def _encrypt_with_algorithm(
        self, 
        data: bytes, 
        key: bytes, 
        algorithm: EncryptionAlgorithm
    ) -> Dict[str, Any]:
        """Encrypt data with specific algorithm"""
        try:
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                result = self.cipher_engine.encrypt_aes_gcm(data, key)
                import base64
                return {
                    "success": True,
                    "ciphertext": result["ciphertext"],
                    "iv": result["nonce"],
                    "iv_b64": base64.b64encode(result["nonce"]).decode(),
                    "tag_b64": ""  # Included in ciphertext for AESGCM
                }
            
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                result = self.cipher_engine.encrypt_chacha20_poly1305(data, key)
                import base64
                return {
                    "success": True,
                    "ciphertext": result["ciphertext"],
                    "iv": result["nonce"],
                    "iv_b64": base64.b64encode(result["nonce"]).decode(),
                    "tag_b64": ""  # Included in ciphertext
                }
            
            elif algorithm == EncryptionAlgorithm.FERNET:
                ciphertext = self.cipher_engine.encrypt_fernet(data, key)
                return {
                    "success": True,
                    "ciphertext": ciphertext,
                    "iv": b"",
                    "iv_b64": "",
                    "tag_b64": ""
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported encryption algorithm: {algorithm}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _decrypt_with_algorithm(
        self, 
        encrypted_record: EncryptedData, 
        key: bytes, 
        algorithm: EncryptionAlgorithm
    ) -> Dict[str, Any]:
        """Decrypt data with specific algorithm"""
        try:
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                data = self.cipher_engine.decrypt_aes_gcm(
                    encrypted_record.encrypted_content,
                    key,
                    encrypted_record.initialization_vector
                )
                return {"success": True, "data": data}
            
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                data = self.cipher_engine.decrypt_chacha20_poly1305(
                    encrypted_record.encrypted_content,
                    key,
                    encrypted_record.initialization_vector
                )
                return {"success": True, "data": data}
            
            elif algorithm == EncryptionAlgorithm.FERNET:
                data = self.cipher_engine.decrypt_fernet(
                    encrypted_record.encrypted_content,
                    key
                )
                return {"success": True, "data": data}
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported decryption algorithm: {algorithm}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_encrypted_data(
        self, 
        request: EncryptionRequest, 
        encryption_result: Dict[str, Any], 
        content_hash: str,
        compression_used: bool
    ) -> str:
        """Store encrypted data in database"""
        async with db_manager.session() as session:
            # Calculate expiration
            expires_at = None
            if request.expires_in_days:
                expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)
            
            encrypted_data = EncryptedData(
                key_id=request.key_id,
                algorithm=request.algorithm.value,
                encrypted_content=encryption_result["ciphertext"],
                initialization_vector=encryption_result.get("iv", b""),
                authentication_tag=encryption_result.get("tag", b""),
                content_type=request.content_type,
                content_hash=content_hash,
                encryption_context={
                    **request.encryption_context,
                    "compression_used": compression_used
                },
                owner_id=request.encryption_context.get("owner_id", "system"),
                privacy_level=request.privacy_level.value,
                expires_at=expires_at
            )
            
            session.add(encrypted_data)
            session.commit()
            session.refresh(encrypted_data)
            
            return str(encrypted_data.data_id)
    
    async def _get_encrypted_data(self, data_id: str) -> Optional[EncryptedData]:
        """Retrieve encrypted data from database"""
        async with db_manager.session() as session:
            return session.query(EncryptedData).filter(
                EncryptedData.data_id == data_id
            ).first()
    
    async def _update_access_timestamp(self, data_id: str):
        """Update last accessed timestamp"""
        async with db_manager.session() as session:
            encrypted_data = session.query(EncryptedData).filter(
                EncryptedData.data_id == data_id
            ).first()
            
            if encrypted_data:
                encrypted_data.accessed_at = datetime.now(timezone.utc)
                session.commit()


# Global encryption service instance
_encryption_service: Optional[EncryptionService] = None

async def get_encryption_service() -> EncryptionService:
    """Get or create the global encryption service instance"""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
        await _encryption_service.initialize()
    return _encryption_service