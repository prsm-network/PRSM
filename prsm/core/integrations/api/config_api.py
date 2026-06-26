"""
Configuration Management API
============================

FastAPI endpoints for managing integration layer configuration and credentials.
Provides secure access to credential storage and configuration management.

Key Endpoints:
- Credential management (store, retrieve, update, delete)
- Configuration management (preferences, platform settings)
- Validation and health checking
- Import/export capabilities
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import Field, SecretStr

from ..config.credential_manager import (
    credential_manager, CredentialData, CredentialType
)
from ..config.integration_config import (
    config_manager, IntegrationPreferences, PlatformConfig,
    SecurityConfig, RateLimitConfig
)
from ..models.integration_models import IntegrationPlatform
from prsm.core.models import PRSMBaseModel


# === API Models ===

class CredentialStoreRequest(PRSMBaseModel):
    """Request model for storing credentials"""
    platform: IntegrationPlatform
    credential_type: str = CredentialType.API_KEY
    api_key: Optional[SecretStr] = None
    access_token: Optional[SecretStr] = None
    refresh_token: Optional[SecretStr] = None
    client_id: Optional[str] = None
    client_secret: Optional[SecretStr] = None
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    expires_in_days: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CredentialResponse(PRSMBaseModel):
    """Response model for credential operations"""
    credential_id: str
    platform: str
    credential_type: str
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    metadata: Dict[str, Any]


class ConfigurationUpdateRequest(PRSMBaseModel):
    """Request model for configuration updates"""
    preferences: Optional[IntegrationPreferences] = None
    global_security: Optional[SecurityConfig] = None
    global_rate_limit: Optional[RateLimitConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlatformConfigRequest(PRSMBaseModel):
    """Request model for platform configuration"""
    enabled: bool = True
    api_base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_concurrent_requests: int = Field(default=5, ge=1, le=20)
    oauth_scopes: List[str] = Field(default_factory=list)
    rate_limit: Optional[RateLimitConfig] = None
    security: Optional[SecurityConfig] = None
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


# === API Router ===

config_router = APIRouter(
    prefix="/config",
    tags=["configuration"],
    responses={404: {"description": "Not found"}}
)


# === Dependency Functions ===

_config_security = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_config_security),
) -> str:
    """Authenticate via the canonical JWT verifier; return the user id as a string.

    sp1278 (audit round 7): was a stub returning "default_user", leaving this credential-vault
    router (store/list/validate/DELETE credentials, settings import/export) effectively
    UNAUTHENTICATED — same pattern fixed in sp1272/sp1268 for the sibling integration routers.
    jwt_handler.verify_token enforces signature, expiry, required claims, and revocation; we
    additionally reject non-access tokens.
    """
    from prsm.core.auth.jwt_handler import jwt_handler

    try:
        token_data = await jwt_handler.verify_token(credentials.credentials)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 — never authenticate on a verifier error
        raise HTTPException(status_code=401, detail="Authentication failed")

    if token_data is None or token_data.token_type != "access":
        raise HTTPException(status_code=401, detail="Invalid or missing access token")
    return str(token_data.user_id)


# === Credential Management Endpoints ===

@config_router.post("/credentials")
async def store_credential(
    request: CredentialStoreRequest,
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Store encrypted credential for platform
    """
    try:
        # Create credential data
        credential_data = CredentialData(
            api_key=request.api_key,
            access_token=request.access_token,
            refresh_token=request.refresh_token,
            client_id=request.client_id,
            client_secret=request.client_secret,
            username=request.username,
            password=request.password,
            custom_fields=request.custom_fields
        )
        
        # Store credential
        credential_id = credential_manager.store_credential(
            user_id=user_id,
            platform=request.platform,
            credential_data=credential_data,
            credential_type=request.credential_type,
            expires_in_days=request.expires_in_days,
            metadata=request.metadata
        )
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Credential stored successfully",
                "credential_id": credential_id,
                "platform": request.platform
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store credential: {str(e)}"
        )


@config_router.get("/credentials")
async def list_credentials(
    platform: Optional[IntegrationPlatform] = None,
    include_expired: bool = False,
    user_id: str = Depends(get_current_user)
) -> List[CredentialResponse]:
    """
    List user's stored credentials (metadata only)
    """
    try:
        credentials = credential_manager.list_credentials(
            user_id=user_id,
            platform=platform,
            include_expired=include_expired
        )
        
        return [
            CredentialResponse(
                credential_id=cred["credential_id"],
                platform=cred["platform"],
                credential_type=cred["credential_type"],
                created_at=datetime.fromisoformat(cred["created_at"]),
                expires_at=datetime.fromisoformat(cred["expires_at"]) if cred["expires_at"] else None,
                is_active=cred["is_active"],
                metadata=cred["metadata"]
            )
            for cred in credentials
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list credentials: {str(e)}"
        )


@config_router.get("/credentials/{credential_id}/validate")
async def validate_credential(
    credential_id: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate specific credential
    """
    try:
        # Find platform for credential
        credentials = credential_manager.list_credentials(user_id)
        target_cred = next((c for c in credentials if c["credential_id"] == credential_id), None)
        
        if not target_cred:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Credential not found"
            )
        
        platform = IntegrationPlatform(target_cred["platform"])
        
        validation_result = credential_manager.validate_credential(
            user_id=user_id,
            platform=platform,
            credential_id=credential_id
        )
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate credential: {str(e)}"
        )


@config_router.delete("/credentials/{credential_id}")
async def delete_credential(
    credential_id: str,
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Delete stored credential
    """
    try:
        success = credential_manager.delete_credential(
            credential_id=credential_id,
            user_id=user_id
        )
        
        if success:
            return JSONResponse(
                content={"message": "Credential deleted successfully"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Credential not found or access denied"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete credential: {str(e)}"
        )


@config_router.post("/credentials/cleanup")
async def cleanup_expired_credentials(
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Clean up expired credentials
    """
    try:
        removed_count = credential_manager.cleanup_expired_credentials()
        
        return JSONResponse(
            content={
                "message": f"Cleaned up {removed_count} expired credentials",
                "removed_count": removed_count
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup credentials: {str(e)}"
        )


# === Configuration Management Endpoints ===

@config_router.get("/settings")
async def get_user_configuration(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user's integration configuration
    """
    try:
        config = config_manager.get_user_config(user_id)
        return config.model_dump()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration: {str(e)}"
        )


@config_router.put("/settings")
async def update_user_configuration(
    request: ConfigurationUpdateRequest,
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Update user's integration configuration
    """
    try:
        success = config_manager.update_user_config(
            user_id=user_id,
            preferences=request.preferences,
            global_security=request.global_security,
            global_rate_limit=request.global_rate_limit,
            metadata=request.metadata
        )
        
        if success:
            return JSONResponse(
                content={"message": "Configuration updated successfully"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update configuration"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )


@config_router.get("/settings/platforms/{platform}")
async def get_platform_configuration(
    platform: IntegrationPlatform,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get platform-specific configuration
    """
    try:
        platform_config = config_manager.get_platform_config(user_id, platform)
        return platform_config.model_dump()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get platform configuration: {str(e)}"
        )


@config_router.put("/settings/platforms/{platform}")
async def update_platform_configuration(
    platform: IntegrationPlatform,
    request: PlatformConfigRequest,
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Update platform-specific configuration
    """
    try:
        # Create platform config object
        platform_config = PlatformConfig(
            platform=platform,
            enabled=request.enabled,
            api_base_url=request.api_base_url,
            api_version=request.api_version,
            timeout_seconds=request.timeout_seconds,
            max_concurrent_requests=request.max_concurrent_requests,
            oauth_scopes=request.oauth_scopes,
            rate_limit=request.rate_limit or RateLimitConfig(),
            security=request.security or SecurityConfig(),
            custom_settings=request.custom_settings
        )
        
        success = config_manager.update_platform_config(
            user_id=user_id,
            platform=platform,
            platform_config=platform_config
        )
        
        if success:
            return JSONResponse(
                content={"message": f"{platform.value} configuration updated successfully"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update platform configuration"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update platform configuration: {str(e)}"
        )


@config_router.get("/settings/validate")
async def validate_configuration(
    platform: Optional[IntegrationPlatform] = None,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate user configuration
    """
    try:
        validation_result = config_manager.validate_configuration(
            user_id=user_id,
            platform=platform
        )
        
        return validation_result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate configuration: {str(e)}"
        )


@config_router.get("/settings/export")
async def export_configuration(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Export user configuration for backup
    """
    try:
        export_data = config_manager.export_user_config(user_id)
        
        if not export_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No configuration found to export"
            )
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export configuration: {str(e)}"
        )


@config_router.post("/settings/import")
async def import_configuration(
    config_data: Dict[str, Any],
    merge: bool = True,
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Import user configuration from backup
    """
    try:
        success = config_manager.import_user_config(
            user_id=user_id,
            config_data=config_data,
            merge=merge
        )
        
        if success:
            return JSONResponse(
                content={
                    "message": f"Configuration imported successfully (merge={merge})"
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to import configuration"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import configuration: {str(e)}"
        )


@config_router.post("/settings/reset")
async def reset_configuration(
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Reset user configuration to defaults
    """
    try:
        success = config_manager.reset_user_config(user_id)
        
        if success:
            return JSONResponse(
                content={"message": "Configuration reset to defaults"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset configuration"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset configuration: {str(e)}"
        )


# === System Information Endpoints ===

@config_router.get("/stats")
async def get_system_statistics() -> Dict[str, Any]:
    """
    Get configuration and credential system statistics
    """
    try:
        config_stats = config_manager.get_system_stats()
        credential_stats = credential_manager.get_storage_stats()
        
        return {
            "configuration": config_stats,
            "credentials": credential_stats,
            "system_info": {
                "encryption_enabled": True,
                "storage_encrypted": True,
                "version": "1.0"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system statistics: {str(e)}"
        )


@config_router.get("/health")
async def get_config_health() -> Dict[str, Any]:
    """
    Get configuration system health status
    """
    try:
        # Test credential storage
        credential_health = True
        try:
            credential_manager.get_storage_stats()
        except Exception:
            credential_health = False
        
        # Test configuration storage
        config_health = True
        try:
            config_manager.get_system_stats()
        except Exception:
            config_health = False
        
        overall_status = "healthy" if (credential_health and config_health) else "degraded"
        
        return {
            "status": overall_status,
            "components": {
                "credential_storage": "healthy" if credential_health else "error",
                "configuration_storage": "healthy" if config_health else "error",
                "encryption": "healthy"
            },
            "last_check": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }