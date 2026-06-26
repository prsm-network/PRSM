"""
Credential Manager
==================

Secure credential storage and management system for PRSM integration layer.
Handles encryption, storage, and retrieval of API keys and authentication
tokens for external platform integrations.

Features:
- Encrypted credential storage using Fernet (AES 128)
- Per-user credential isolation and management
- Automatic credential rotation and expiry handling
- Support for multiple credential types (API keys, OAuth tokens, etc.)
- Secure credential validation and health checking
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

from cryptography.fernet import Fernet
import base64
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_serializer

from ..models.integration_models import IntegrationPlatform
from prsm.core.config import settings


class CredentialType(str):
    """Types of credentials supported"""
    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    CUSTOM = "custom"


class StoredCredential(BaseModel):
    """Model for stored credential data"""
    credential_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    platform: IntegrationPlatform
    credential_type: str
    encrypted_data: bytes
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CredentialData(BaseModel):
    """Model for decrypted credential data"""
    api_key: Optional[SecretStr] = None
    access_token: Optional[SecretStr] = None
    refresh_token: Optional[SecretStr] = None
    client_id: Optional[str] = None
    client_secret: Optional[SecretStr] = None
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # Sprint 338 — closed the V3-prep arc on this file. The
    # SecretStr fields get a JSON-only field_serializer that
    # reveals the underlying secret when the model is dumped
    # to JSON (the legacy `json_encoders` behavior). The same
    # serializer covers every SecretStr field on the model;
    # `when_used="json"` keeps `.model_dump()` (Python dict)
    # behavior unchanged (returns SecretStr objects).
    @field_serializer(
        "api_key",
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        when_used="json",
    )
    def _dump_secret(
        self, v: Optional[SecretStr],
    ) -> Optional[str]:
        return v.get_secret_value() if v is not None else None


class CredentialManager:
    """
    Secure credential management system for integration layer
    
    Provides encrypted storage and retrieval of API keys, OAuth tokens,
    and other authentication credentials for external platform integrations.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize credential manager"""
        
        # Storage configuration
        self.storage_dir = Path(storage_dir or getattr(settings, "credential_storage_dir", "~/.prsm/credentials")).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Encryption setup
        self.master_key_file = self.storage_dir / "master.key"
        self.credentials_file = self.storage_dir / "credentials.json"
        
        # In-memory storage for active credentials
        self.active_credentials: Dict[str, StoredCredential] = {}
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Load existing credentials
        self._load_credentials()
        
        print(f"🔐 Credential manager initialized with storage at {self.storage_dir}")
    
    def _initialize_encryption(self):
        """Initialize encryption system"""
        try:
            if self.master_key_file.exists():
                # Load existing master key
                with open(self.master_key_file, 'rb') as f:
                    self.master_key = f.read()
            else:
                # Generate new master key
                self.master_key = Fernet.generate_key()

                # sp1281 — atomic 0o600 write (no world-readable window between create + chmod;
                # this is the Fernet master key that decrypts every stored credential).
                from prsm.node.identity import write_owner_only_file
                write_owner_only_file(self.master_key_file, self.master_key)

                print("🔑 Generated new master encryption key")
            
            self.cipher = Fernet(self.master_key)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize encryption: {e}")
    
    def _load_credentials(self):
        """Load credentials from storage"""
        try:
            if not self.credentials_file.exists():
                return
            
            with open(self.credentials_file, 'r') as f:
                data = json.load(f)
            
            for cred_data in data.get('credentials', []):
                try:
                    # Convert encrypted_data back to bytes
                    cred_data['encrypted_data'] = base64.b64decode(cred_data['encrypted_data'])
                    
                    # Parse dates
                    for date_field in ['created_at', 'updated_at', 'expires_at', 'last_used_at']:
                        if cred_data.get(date_field):
                            cred_data[date_field] = datetime.fromisoformat(cred_data[date_field].replace('Z', '+00:00'))
                    
                    credential = StoredCredential(**cred_data)
                    self.active_credentials[credential.credential_id] = credential
                    
                except Exception as e:
                    print(f"⚠️ Failed to load credential {cred_data.get('credential_id', 'unknown')}: {e}")
            
            print(f"📋 Loaded {len(self.active_credentials)} stored credentials")
            
        except Exception as e:
            print(f"⚠️ Failed to load credentials: {e}")
    
    def _save_credentials(self):
        """Save credentials to storage"""
        try:
            data = {
                'version': '1.0',
                'credentials': []
            }
            
            for credential in self.active_credentials.values():
                cred_data = credential.model_dump()
                
                # Convert bytes to base64 for JSON serialization
                cred_data['encrypted_data'] = base64.b64encode(cred_data['encrypted_data']).decode('utf-8')
                
                # Convert dates to ISO format
                for date_field in ['created_at', 'updated_at', 'expires_at', 'last_used_at']:
                    if cred_data.get(date_field):
                        cred_data[date_field] = cred_data[date_field].isoformat()
                
                data['credentials'].append(cred_data)
            
            # Write atomically
            temp_file = self.credentials_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.credentials_file)
            
            # Set restrictive permissions
            os.chmod(self.credentials_file, 0o600)
            
        except Exception as e:
            print(f"❌ Failed to save credentials: {e}")
            raise
    
    def _encrypt_credential_data(self, data: CredentialData) -> bytes:
        """Encrypt credential data"""
        try:
            json_data = data.model_dump_json()
            encrypted_data = self.cipher.encrypt(json_data.encode('utf-8'))
            return encrypted_data
            
        except Exception as e:
            raise ValueError(f"Failed to encrypt credential data: {e}")
    
    def _decrypt_credential_data(self, encrypted_data: bytes) -> CredentialData:
        """Decrypt credential data"""
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
            json_data = json.loads(decrypted_data.decode('utf-8'))
            return CredentialData(**json_data)
            
        except Exception as e:
            raise ValueError(f"Failed to decrypt credential data: {e}")
    
    # === Public API ===
    
    def store_credential(
        self,
        user_id: str,
        platform: IntegrationPlatform,
        credential_data: CredentialData,
        credential_type: str = CredentialType.API_KEY,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store encrypted credential for user and platform
        
        Args:
            user_id: User identifier
            platform: Integration platform
            credential_data: Credential information to encrypt
            credential_type: Type of credential
            expires_in_days: Optional expiration in days
            metadata: Additional metadata
            
        Returns:
            Credential ID for future reference
        """
        try:
            # Calculate expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
            
            # Encrypt credential data
            encrypted_data = self._encrypt_credential_data(credential_data)
            
            # Create stored credential
            credential = StoredCredential(
                user_id=user_id,
                platform=platform,
                credential_type=credential_type,
                encrypted_data=encrypted_data,
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            # Remove any existing credentials for this user/platform combination
            existing_ids = [
                cred_id for cred_id, cred in self.active_credentials.items()
                if cred.user_id == user_id and cred.platform == platform
            ]
            for cred_id in existing_ids:
                del self.active_credentials[cred_id]
            
            # Store new credential
            self.active_credentials[credential.credential_id] = credential
            
            # Save to disk
            self._save_credentials()
            
            print(f"🔐 Stored {credential_type} for {user_id} on {platform.value}")
            return credential.credential_id
            
        except Exception as e:
            print(f"❌ Failed to store credential: {e}")
            raise
    
    def get_credential(
        self,
        user_id: str,
        platform: IntegrationPlatform,
        credential_id: Optional[str] = None
    ) -> Optional[CredentialData]:
        """
        Retrieve and decrypt credential for user and platform
        
        Args:
            user_id: User identifier
            platform: Integration platform
            credential_id: Optional specific credential ID
            
        Returns:
            Decrypted credential data or None if not found
        """
        try:
            # Find matching credential
            target_credential = None
            
            if credential_id:
                # Get specific credential
                credential = self.active_credentials.get(credential_id)
                if credential and credential.user_id == user_id and credential.platform == platform:
                    target_credential = credential
            else:
                # Get most recent active credential for user/platform
                matching_credentials = [
                    cred for cred in self.active_credentials.values()
                    if (cred.user_id == user_id and 
                        cred.platform == platform and 
                        cred.is_active and
                        (not cred.expires_at or cred.expires_at > datetime.now(timezone.utc)))
                ]
                
                if matching_credentials:
                    # Sort by most recent
                    target_credential = max(matching_credentials, key=lambda c: c.updated_at)
            
            if not target_credential:
                return None
            
            # Update last used time
            target_credential.last_used_at = datetime.now(timezone.utc)
            self._save_credentials()
            
            # Decrypt and return
            decrypted_data = self._decrypt_credential_data(target_credential.encrypted_data)
            return decrypted_data
            
        except Exception as e:
            print(f"❌ Failed to retrieve credential: {e}")
            return None
    
    def list_credentials(
        self,
        user_id: str,
        platform: Optional[IntegrationPlatform] = None,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List credentials for user (metadata only, no decryption)
        
        Args:
            user_id: User identifier
            platform: Optional platform filter
            include_expired: Whether to include expired credentials
            
        Returns:
            List of credential metadata
        """
        try:
            credentials = []
            now = datetime.now(timezone.utc)
            
            for credential in self.active_credentials.values():
                # Filter by user
                if credential.user_id != user_id:
                    continue
                
                # Filter by platform
                if platform and credential.platform != platform:
                    continue
                
                # Filter expired
                if not include_expired and credential.expires_at and credential.expires_at <= now:
                    continue
                
                # Add metadata (no sensitive data)
                cred_info = {
                    'credential_id': credential.credential_id,
                    'platform': credential.platform,
                    'credential_type': credential.credential_type,
                    'created_at': credential.created_at.isoformat(),
                    'updated_at': credential.updated_at.isoformat(),
                    'expires_at': credential.expires_at.isoformat() if credential.expires_at else None,
                    'last_used_at': credential.last_used_at.isoformat() if credential.last_used_at else None,
                    'is_active': credential.is_active,
                    'is_expired': credential.expires_at and credential.expires_at <= now,
                    'metadata': credential.metadata
                }
                
                credentials.append(cred_info)
            
            # Sort by most recent
            credentials.sort(key=lambda c: c['updated_at'], reverse=True)
            return credentials
            
        except Exception as e:
            print(f"❌ Failed to list credentials: {e}")
            return []
    
    def update_credential(
        self,
        credential_id: str,
        user_id: str,
        credential_data: Optional[CredentialData] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_in_days: Optional[int] = None
    ) -> bool:
        """
        Update existing credential
        
        Args:
            credential_id: Credential identifier
            user_id: User identifier (for security)
            credential_data: New credential data (optional)
            metadata: New metadata (optional)
            expires_in_days: New expiration in days (optional)
            
        Returns:
            True if updated successfully
        """
        try:
            credential = self.active_credentials.get(credential_id)
            if not credential or credential.user_id != user_id:
                return False
            
            # Update credential data if provided
            if credential_data:
                credential.encrypted_data = self._encrypt_credential_data(credential_data)
            
            # Update metadata if provided
            if metadata:
                credential.metadata.update(metadata)
            
            # Update expiration if provided
            if expires_in_days is not None:
                if expires_in_days > 0:
                    credential.expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
                else:
                    credential.expires_at = None
            
            # Update timestamp
            credential.updated_at = datetime.now(timezone.utc)
            
            # Save changes
            self._save_credentials()
            
            print(f"🔄 Updated credential {credential_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update credential: {e}")
            return False
    
    def delete_credential(
        self,
        credential_id: str,
        user_id: str
    ) -> bool:
        """
        Delete credential
        
        Args:
            credential_id: Credential identifier
            user_id: User identifier (for security)
            
        Returns:
            True if deleted successfully
        """
        try:
            credential = self.active_credentials.get(credential_id)
            if not credential or credential.user_id != user_id:
                return False
            
            # Remove from memory
            del self.active_credentials[credential_id]
            
            # Save changes
            self._save_credentials()
            
            print(f"🗑️ Deleted credential {credential_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete credential: {e}")
            return False
    
    def validate_credential(
        self,
        user_id: str,
        platform: IntegrationPlatform,
        credential_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate credential without exposing sensitive data
        
        Args:
            user_id: User identifier
            platform: Integration platform
            credential_id: Optional specific credential ID
            
        Returns:
            Validation result with status and metadata
        """
        try:
            # Get credential metadata (include expired to properly report status)
            credentials = self.list_credentials(user_id, platform, include_expired=True)
            
            if credential_id:
                credentials = [c for c in credentials if c['credential_id'] == credential_id]
            
            if not credentials:
                return {
                    'valid': False,
                    'status': 'not_found',
                    'message': 'No credential found'
                }
            
            credential = credentials[0]  # Most recent
            
            # Check expiration
            if credential['is_expired']:
                return {
                    'valid': False,
                    'status': 'expired',
                    'message': 'Credential has expired',
                    'expires_at': credential['expires_at']
                }
            
            # Check if active
            if not credential['is_active']:
                return {
                    'valid': False,
                    'status': 'inactive',
                    'message': 'Credential is inactive'
                }
            
            return {
                'valid': True,
                'status': 'valid',
                'message': 'Credential is valid',
                'credential_id': credential['credential_id'],
                'credential_type': credential['credential_type'],
                'last_used_at': credential['last_used_at'],
                'expires_at': credential['expires_at']
            }
            
        except Exception as e:
            return {
                'valid': False,
                'status': 'error',
                'message': f'Validation failed: {str(e)}'
            }
    
    def cleanup_expired_credentials(self) -> int:
        """
        Remove expired credentials from storage
        
        Returns:
            Number of credentials removed
        """
        try:
            now = datetime.now(timezone.utc)
            expired_ids = []
            
            for cred_id, credential in self.active_credentials.items():
                if credential.expires_at and credential.expires_at <= now:
                    expired_ids.append(cred_id)
            
            # Remove expired credentials
            for cred_id in expired_ids:
                del self.active_credentials[cred_id]
            
            if expired_ids:
                self._save_credentials()
                print(f"🧹 Cleaned up {len(expired_ids)} expired credentials")
            
            return len(expired_ids)
            
        except Exception as e:
            print(f"❌ Failed to cleanup expired credentials: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get credential storage statistics"""
        try:
            now = datetime.now(timezone.utc)
            total_credentials = len(self.active_credentials)
            
            active_credentials = sum(1 for c in self.active_credentials.values() if c.is_active)
            expired_credentials = sum(
                1 for c in self.active_credentials.values()
                if c.expires_at and c.expires_at <= now
            )
            
            platforms = {}
            credential_types = {}
            users = set()
            
            for credential in self.active_credentials.values():
                platforms[credential.platform] = platforms.get(credential.platform, 0) + 1
                credential_types[credential.credential_type] = credential_types.get(credential.credential_type, 0) + 1
                users.add(credential.user_id)
            
            return {
                'total_credentials': total_credentials,
                'active_credentials': active_credentials,
                'expired_credentials': expired_credentials,
                'unique_users': len(users),
                'platforms': platforms,
                'credential_types': credential_types,
                'storage_path': str(self.storage_dir)
            }
            
        except Exception as e:
            print(f"❌ Failed to get storage stats: {e}")
            return {}


# Global credential manager instance
credential_manager = CredentialManager()