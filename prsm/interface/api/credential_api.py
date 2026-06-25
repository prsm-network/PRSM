"""
Credential Management API
=========================

REST API endpoints for secure credential management.
Provides authenticated access to credential registration,
validation, rotation, and monitoring.
"""

import structlog
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from prsm.core.auth import get_current_user
from prsm.core.auth.models import UserRole
from prsm.core.security.enhanced_authorization import get_enhanced_auth_manager
from prsm.core.integrations.security.secure_api_client_factory import SecureClientType, secure_client_factory
from prsm.core.integrations.security.secure_config_manager import secure_config_manager
from prsm.core.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/credentials", tags=["credentials"])


class CredentialRegistrationRequest(BaseModel):
    """Request model for credential registration"""
    platform: str = Field(..., description="Platform name (openai, anthropic, huggingface, etc.)")
    credentials: Dict[str, Any] = Field(..., description="Credential data")
    expires_at: Optional[datetime] = Field(None, description="Optional expiration time")
    user_specific: bool = Field(False, description="Whether these are user-specific credentials")


class CredentialValidationRequest(BaseModel):
    """Request model for credential validation"""
    platform: str = Field(..., description="Platform name to validate")
    user_specific: bool = Field(False, description="Whether to check user-specific credentials")


class CredentialRotationRequest(BaseModel):
    """Request model for credential rotation"""
    platform: str = Field(..., description="Platform name to rotate credentials for")
    user_specific: bool = Field(False, description="Whether to rotate user-specific credentials")


class CredentialStatusResponse(BaseModel):
    """Response model for credential status"""
    platform: str
    credentials_available: bool
    last_validated: Optional[datetime]
    expires_at: Optional[datetime]
    secure: bool


class SystemCredentialStatusResponse(BaseModel):
    """Response model for system credential status"""
    migration_completed: bool
    system_secrets_secure: bool
    platform_credentials: Dict[str, Dict[str, Any]]
    credential_manager_available: bool
    total_platforms: int
    platforms_with_credentials: int


@router.post("/register")
async def register_credentials(
    request: CredentialRegistrationRequest,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Register API credentials for external service integration
    
    🔐 SECURITY:
    - Requires authentication
    - Credentials are encrypted before storage
    - Audit logging for all credential registration
    - User-specific or system-level credential storage
    """
    try:
        # Determine user ID for credential storage
        user_id = current_user if request.user_specific else "system"
        
        # Validate platform
        platform_lower = request.platform.lower()
        valid_platforms = [e.value for e in SecureClientType]
        
        if platform_lower not in valid_platforms:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid platform. Must be one of: {', '.join(valid_platforms)}"
            )
        
        # Register credentials
        success = await secure_config_manager.register_api_credentials(
            platform=platform_lower,
            credentials=request.credentials,
            user_id=user_id,
            expires_at=request.expires_at
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to register credentials"
            )
        
        # Log successful registration
        await audit_logger.log_security_event(
            event_type="credential_registration_success",
            user_id=current_user,
            details={
                "platform": platform_lower,
                "user_specific": request.user_specific,
                "expires_at": request.expires_at.isoformat() if request.expires_at else None
            },
            security_level="info"
        )
        
        return {
            "success": True,
            "message": f"Credentials registered successfully for {request.platform}",
            "platform": platform_lower,
            "user_specific": request.user_specific,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to register credentials",
                    platform=request.platform,
                    user_id=current_user,
                    error=str(e))
        
        await audit_logger.log_security_event(
            event_type="credential_registration_failed",
            user_id=current_user,
            details={
                "platform": request.platform,
                "error": str(e)
            },
            security_level="error"
        )
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error during credential registration"
        )


@router.post("/validate")
async def validate_credentials(
    request: CredentialValidationRequest,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate that credentials are available and valid for a platform
    
    🔐 SECURITY:
    - Requires authentication
    - Does not return actual credential values
    - Validates credential integrity and availability
    """
    try:
        # Determine user ID for validation
        user_id = current_user if request.user_specific else "system"
        
        # Map platform to client type
        platform_lower = request.platform.lower()
        try:
            client_type = SecureClientType(platform_lower)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid platform: {request.platform}"
            )
        
        # Validate credentials
        is_valid = await secure_client_factory.validate_client_credentials(
            client_type, user_id
        )
        
        return {
            "success": True,
            "platform": platform_lower,
            "credentials_available": is_valid,
            "user_specific": request.user_specific,
            "validated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to validate credentials",
                    platform=request.platform,
                    user_id=current_user,
                    error=str(e))
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error during credential validation"
        )


@router.post("/rotate")
async def rotate_credentials(
    request: CredentialRotationRequest,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Rotate credentials for a platform (where supported)
    
    🔐 SECURITY:
    - Requires authentication
    - Logs all rotation attempts
    - Platform-specific rotation logic
    """
    try:
        # Determine user ID for rotation
        user_id = current_user if request.user_specific else "system"
        
        # Rotate credentials
        success = await secure_config_manager.rotate_platform_credentials(
            platform=request.platform.lower(),
            user_id=user_id
        )
        
        # Log rotation attempt
        await audit_logger.log_security_event(
            event_type="credential_rotation_attempted",
            user_id=current_user,
            details={
                "platform": request.platform,
                "success": success,
                "user_specific": request.user_specific
            },
            security_level="info"
        )
        
        return {
            "success": success,
            "platform": request.platform.lower(),
            "message": "Credential rotation completed" if success else "Credential rotation not supported for this platform",
            "user_specific": request.user_specific,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to rotate credentials",
                    platform=request.platform,
                    user_id=current_user,
                    error=str(e))
        
        await audit_logger.log_security_event(
            event_type="credential_rotation_failed",
            user_id=current_user,
            details={
                "platform": request.platform,
                "error": str(e)
            },
            security_level="error"
        )
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error during credential rotation"
        )


@router.get("/status")
async def get_credential_status(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get credential status for current user
    
    🔐 SECURITY:
    - Requires authentication
    - Returns only availability status, not actual credentials
    - Shows both user-specific and system credentials
    """
    try:
        user_credential_status = {}
        system_credential_status = {}
        
        # Check all supported platforms
        for client_type in SecureClientType:
            # Check user-specific credentials
            user_has_creds = await secure_client_factory.validate_client_credentials(
                client_type, current_user
            )
            
            # Check system credentials
            system_has_creds = await secure_client_factory.validate_client_credentials(
                client_type, "system"
            )
            
            user_credential_status[client_type.value] = {
                "credentials_available": user_has_creds,
                "last_checked": datetime.now(timezone.utc).isoformat()
            }
            
            system_credential_status[client_type.value] = {
                "credentials_available": system_has_creds,
                "last_checked": datetime.now(timezone.utc).isoformat()
            }
        
        return {
            "success": True,
            "user_id": current_user,
            "user_credentials": user_credential_status,
            "system_credentials": system_credential_status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get credential status",
                    user_id=current_user,
                    error=str(e))
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error getting credential status"
        )


@router.get("/system/status")
async def get_system_credential_status(
    current_user: str = Depends(get_current_user)
) -> SystemCredentialStatusResponse:
    """
    Get system-wide credential status (admin only)
    
    🔐 SECURITY:
    - Requires authentication and admin permissions
    - Returns comprehensive system credential information
    - Used for system monitoring and maintenance
    """
    try:
        # Check admin permissions for credential system access
        auth_manager = get_enhanced_auth_manager()
        has_permission = await auth_manager.check_permission(
            user_id=str(getattr(current_user, "id", current_user)),
            # sp1264 — use the authenticated user's REAL role; hardcoding UserRole.ADMIN
            # short-circuited check_permission to True for ANY authenticated caller.
            user_role=getattr(current_user, "role", UserRole.USER),
            resource_type="system_credentials",
            action="read"
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role required for credential system access"
            )
        
        # Get secure configuration status. NB: must NOT be named `status` — that shadows the
        # imported FastAPI `status` module, so `status.HTTP_403_FORBIDDEN` in the deny branch
        # above would UnboundLocalError (this was masked while the hardcoded-ADMIN bypass kept
        # has_permission always True; sp1264 made the deny branch reachable).
        config_status = await secure_config_manager.get_secure_configuration_status()

        # Calculate statistics
        platform_count = len(config_status.get("platform_credentials", {}))
        platforms_with_creds = sum(
            1 for creds in config_status.get("platform_credentials", {}).values()
            if creds.get("credentials_available", False)
        )

        return SystemCredentialStatusResponse(
            migration_completed=config_status.get("migration_completed", False),
            system_secrets_secure=config_status.get("system_secrets_secure", False),
            platform_credentials=config_status.get("platform_credentials", {}),
            credential_manager_available=config_status.get("credential_manager_available", False),
            total_platforms=platform_count,
            platforms_with_credentials=platforms_with_creds
        )

    except HTTPException:
        raise  # sp1264 — let the 403 (and any deliberate HTTP error) surface; don't mask it as 500
    except Exception as e:
        logger.error("Failed to get system credential status",
                    user_id=current_user,
                    error=str(e))

        raise HTTPException(
            status_code=500,
            detail="Internal server error getting system credential status"
        )


@router.post("/system/initialize")
async def initialize_secure_configuration_endpoint(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Initialize secure configuration system (admin only)
    
    🔐 SECURITY:
    - Requires authentication and admin permissions
    - Migrates environment variables to encrypted storage
    - Sets up secure configuration templates
    """
    try:
        # Check admin permissions for secure configuration initialization
        auth_manager = get_enhanced_auth_manager()
        has_permission = await auth_manager.check_permission(
            user_id=str(getattr(current_user, "id", current_user)),
            # sp1264 — use the authenticated user's REAL role (see /system/status above).
            user_role=getattr(current_user, "role", UserRole.USER),
            resource_type="secure_configuration",
            action="create"
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role required for secure configuration initialization"
            )
        
        # Initialize secure configuration
        success = await secure_config_manager.initialize_secure_configuration()
        
        # Log initialization attempt
        await audit_logger.log_security_event(
            event_type="secure_config_initialization_requested",
            user_id=current_user,
            details={
                "success": success
            },
            security_level="info" if success else "warning"
        )
        
        return {
            "success": success,
            "message": "Secure configuration initialization completed" if success else "Secure configuration initialization failed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise  # sp1264 — let the 403 surface; don't mask the auth denial as a 500
    except Exception as e:
        logger.error("Failed to initialize secure configuration",
                    user_id=current_user,
                    error=str(e))
        
        await audit_logger.log_security_event(
            event_type="secure_config_initialization_failed",
            user_id=current_user,
            details={
                "error": str(e)
            },
            security_level="error"
        )
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error during secure configuration initialization"
        )


@router.get("/platforms")
async def list_supported_platforms(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all supported platforms for credential management
    
    🔐 SECURITY:
    - Requires authentication
    - Returns platform information and credential requirements
    """
    try:
        platforms = []
        
        for client_type in SecureClientType:
            # Get required fields for each platform
            required_fields = []
            if client_type == SecureClientType.GITHUB:
                required_fields = ["access_token"]
            elif client_type == SecureClientType.PINECONE:
                required_fields = ["api_key", "environment"]
            elif client_type in [SecureClientType.WEAVIATE, SecureClientType.OLLAMA]:
                required_fields = ["url"]
            else:
                required_fields = ["api_key"]
            
            platforms.append({
                "platform": client_type.value,
                "name": client_type.value.title(),
                "required_fields": required_fields,
                "credential_type": "oauth_token" if client_type == SecureClientType.GITHUB else "api_key"
            })
        
        return {
            "success": True,
            "platforms": platforms,
            "total_count": len(platforms),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to list supported platforms",
                    user_id=current_user,
                    error=str(e))
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error listing supported platforms"
        )