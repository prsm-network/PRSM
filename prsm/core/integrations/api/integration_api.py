"""
Integration API Endpoints
========================

FastAPI router for integration layer endpoints, providing REST API
access to platform connectors and import operations.

Key Endpoints:
- Platform connection management
- Content search and discovery
- Import request submission and tracking
- Health and status monitoring
- Integration analytics
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import Field

from prsm.core.integrations.core.integration_manager import integration_manager
from ..connectors.github_connector import GitHubConnector
from ..connectors.huggingface_connector import HuggingFaceConnector
from ..connectors.ollama_connector import OllamaConnector
from ..models.integration_models import (
    IntegrationPlatform, ConnectorConfig, IntegrationSource,
    ImportRequest, ImportStatus, ConnectorHealth,
    IntegrationStats
)
from prsm.core.models import PRSMBaseModel


# === API Models ===

class ConnectorConfigRequest(PRSMBaseModel):
    """Request model for connector configuration"""
    platform: IntegrationPlatform
    oauth_credentials: Optional[Dict[str, str]] = None
    api_key: Optional[str] = None
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(PRSMBaseModel):
    """Request model for content search"""
    query: str
    platforms: Optional[List[IntegrationPlatform]] = None
    content_type: str = "model"
    limit: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None


class ImportRequestModel(PRSMBaseModel):
    """Request model for content import"""
    source: IntegrationSource
    import_type: str = "model"
    target_location: Optional[str] = None
    import_options: Dict[str, Any] = Field(default_factory=dict)
    security_scan_required: bool = True
    license_check_required: bool = True
    auto_reward_creator: bool = True


class ImportStatusResponse(PRSMBaseModel):
    """Response model for import status"""
    request_id: UUID
    status: ImportStatus
    progress_percentage: Optional[float] = None
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    estimated_completion: Optional[datetime] = None


class HealthResponse(PRSMBaseModel):
    """Response model for system health"""
    overall_status: str
    health_percentage: float
    connectors: Dict[str, Any]
    imports: Dict[str, Any]
    last_health_check: Optional[datetime]
    sandbox_status: Dict[str, Any]


# === API Router ===

integration_router = APIRouter(
    prefix="/integrations",
    tags=["integrations"],
    responses={404: {"description": "Not found"}}
)


# === Dependency Functions ===

_integration_security = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_integration_security),
) -> str:
    """Authenticate the caller via the canonical JWT verifier and return their user id.

    sp1272 (audit round 5): was a stub returning "default_user", which left the entire
    /integrations/* router (connector registration, import submission, import history)
    effectively UNAUTHENTICATED. jwt_handler.verify_token enforces signature, expiry, required
    claims, and revocation; we additionally reject non-access tokens. Returns the user id as a
    string to preserve the endpoints' existing contract.
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


async def validate_platform_support(platform: IntegrationPlatform) -> bool:
    """Validate that platform is supported"""
    supported_platforms = [IntegrationPlatform.GITHUB, IntegrationPlatform.HUGGINGFACE, IntegrationPlatform.OLLAMA]
    return platform in supported_platforms


# === Connector Management Endpoints ===

@integration_router.post("/connectors/register")
async def register_connector(
    config_request: ConnectorConfigRequest,
    user_id: str = Depends(get_current_user)
) -> JSONResponse:
    """
    Register and initialize a platform connector
    """
    try:
        if not await validate_platform_support(config_request.platform):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Platform {config_request.platform} not supported"
            )
        
        # Create connector config
        config = ConnectorConfig(
            platform=config_request.platform,
            user_id=user_id,
            oauth_credentials=config_request.oauth_credentials,
            api_key=config_request.api_key,
            custom_settings=config_request.custom_settings
        )
        
        # Select appropriate connector class
        if config_request.platform == IntegrationPlatform.GITHUB:
            connector_class = GitHubConnector
        elif config_request.platform == IntegrationPlatform.HUGGINGFACE:
            connector_class = HuggingFaceConnector
        elif config_request.platform == IntegrationPlatform.OLLAMA:
            connector_class = OllamaConnector
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No connector implementation for {config_request.platform}"
            )
        
        # Register connector
        success = await integration_manager.register_connector(connector_class, config)
        
        if success:
            return JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content={
                    "message": f"{config_request.platform} connector registered successfully",
                    "platform": config_request.platform,
                    "user_id": user_id
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register connector"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Connector registration failed: {str(e)}"
        )


@integration_router.get("/connectors/health")
async def get_connectors_health() -> Dict[IntegrationPlatform, ConnectorHealth]:
    """
    Get health status of all registered connectors
    """
    try:
        health_results = await integration_manager.health_check_all_connectors()
        
        # Convert to serializable format
        return {
            platform.value: health.model_dump() 
            for platform, health in health_results.items()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@integration_router.get("/connectors/{platform}/status")
async def get_connector_status(platform: IntegrationPlatform) -> Dict[str, Any]:
    """
    Get detailed status for specific platform connector
    """
    try:
        connector = await integration_manager.get_connector(platform)
        
        if not connector:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No connector found for platform {platform.value}"
            )
        
        return connector.get_metrics()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get connector status: {str(e)}"
        )


# === Content Discovery Endpoints ===

@integration_router.post("/search")
async def search_content(search_request: SearchRequest) -> List[IntegrationSource]:
    """
    Search for content across integrated platforms
    """
    try:
        results = await integration_manager.search_content(
            query=search_request.query,
            platforms=search_request.platforms,
            content_type=search_request.content_type,
            limit=search_request.limit,
            filters=search_request.filters
        )
        
        # Convert to serializable format
        return [result.model_dump() for result in results]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@integration_router.get("/content/{platform}/{external_id:path}/metadata")
async def get_content_metadata(platform: IntegrationPlatform, external_id: str) -> Dict[str, Any]:
    """
    Get detailed metadata for specific content
    """
    try:
        connector = await integration_manager.get_connector(platform)
        
        if not connector:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No connector available for platform {platform.value}"
            )
        
        metadata = await connector.get_content_metadata(external_id)
        
        if "error" in metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=metadata["error"]
            )
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get content metadata: {str(e)}"
        )


@integration_router.get("/content/{platform}/{external_id:path}/license")
async def validate_content_license(platform: IntegrationPlatform, external_id: str) -> Dict[str, Any]:
    """
    Validate license compliance for specific content
    """
    try:
        connector = await integration_manager.get_connector(platform)
        
        if not connector:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No connector available for platform {platform.value}"
            )
        
        license_info = await connector.validate_license(external_id)
        return license_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"License validation failed: {str(e)}"
        )


# === Import Management Endpoints ===

@integration_router.post("/import")
async def submit_import_request(
    import_request: ImportRequestModel,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Submit content import request
    """
    try:
        # Create import request
        request = ImportRequest(
            user_id=user_id,
            source=import_request.source,
            import_type=import_request.import_type,
            target_location=import_request.target_location,
            import_options=import_request.import_options,
            security_scan_required=import_request.security_scan_required,
            license_check_required=import_request.license_check_required,
            auto_reward_creator=import_request.auto_reward_creator
        )
        
        # Submit to integration manager
        request_id = await integration_manager.submit_import_request(request)
        
        return {
            "request_id": str(request_id),
            "status": "submitted",
            "message": "Import request submitted successfully",
            "source": import_request.source.model_dump()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Import submission failed: {str(e)}"
        )


@integration_router.get("/import/{request_id}/status")
async def get_import_status(request_id: UUID) -> ImportStatusResponse:
    """
    Get status of import operation
    """
    try:
        import_status = await integration_manager.get_import_status(request_id)
        
        if import_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Import request not found"
            )
        
        # Get additional details if available
        result = await integration_manager.get_import_result(request_id)
        error_message = None
        
        if result and result.error_details:
            error_message = result.error_details.get("reason")
        
        return ImportStatusResponse(
            request_id=request_id,
            status=import_status,
            error_message=error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get import status: {str(e)}"
        )


@integration_router.get("/import/{request_id}/result")
async def get_import_result(request_id: UUID) -> Dict[str, Any]:
    """
    Get detailed result of completed import operation
    """
    try:
        result = await integration_manager.get_import_result(request_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Import result not found"
            )
        
        return result.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get import result: {str(e)}"
        )


@integration_router.delete("/import/{request_id}")
async def cancel_import_request(request_id: UUID) -> Dict[str, str]:
    """
    Cancel active import operation
    """
    try:
        success = await integration_manager.cancel_import(request_id)
        
        if success:
            return {"message": "Import request cancelled successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Import request not found or cannot be cancelled"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel import: {str(e)}"
        )


# === System Health and Analytics Endpoints ===

@integration_router.get("/health")
async def get_system_health() -> HealthResponse:
    """
    Get overall integration system health
    """
    try:
        health_data = await integration_manager.get_system_health()
        
        return HealthResponse(
            overall_status=health_data["overall_status"],
            health_percentage=health_data["health_percentage"],
            connectors=health_data["connectors"],
            imports=health_data["imports"],
            last_health_check=health_data["last_health_check"],
            sandbox_status=health_data["sandbox_status"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@integration_router.get("/stats")
async def get_integration_stats() -> IntegrationStats:
    """
    Get comprehensive integration layer statistics
    """
    try:
        stats = await integration_manager.get_integration_stats()
        return stats.model_dump()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get integration stats: {str(e)}"
        )


@integration_router.get("/imports/active")
async def get_active_imports(user_id: str = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """
    Get list of active import operations for user
    """
    try:
        # Get active imports from integration manager
        active_imports = []
        
        for request_id, request in integration_manager.active_imports.items():
            if request.user_id == user_id:
                active_imports.append({
                    "request_id": str(request_id),
                    "source": request.source.model_dump(),
                    "status": request.status.value,
                    "created_at": request.created_at.isoformat(),
                    "import_type": request.import_type
                })
        
        return active_imports
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active imports: {str(e)}"
        )


@integration_router.get("/imports/history")
async def get_import_history(
    user_id: str = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get import history for user
    """
    try:
        # Get import history from integration manager
        user_imports = []
        
        for result in integration_manager.import_history:
            # Find corresponding request
            request = None
            for req_id, req in integration_manager.active_imports.items():
                if req_id == result.request_id:
                    request = req
                    break
            
            if request and request.user_id == user_id:
                user_imports.append({
                    "request_id": str(result.request_id),
                    "result_id": str(result.result_id),
                    "status": result.status.value,
                    "created_at": result.created_at.isoformat(),
                    "completed_at": result.updated_at.isoformat() if result.updated_at else None,
                    "import_duration": result.import_duration,
                    "success_message": result.success_message,
                    "error_details": result.error_details
                })
        
        # Sort by creation time (newest first) and apply pagination
        user_imports.sort(key=lambda x: x["created_at"], reverse=True)
        return user_imports[offset:offset + limit]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get import history: {str(e)}"
        )