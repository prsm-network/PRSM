"""
WebSocket Authentication and Authorization
Secure authentication middleware for PRSM WebSocket connections
"""

import structlog
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from urllib.parse import parse_qs
from uuid import UUID

from fastapi import WebSocket
from pydantic import BaseModel

from prsm.core.auth import jwt_handler
from prsm.core.auth.models import UserRole
from prsm.core.database import get_database_service
from prsm.core.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class WebSocketAuthError(Exception):
    """WebSocket authentication error"""
    def __init__(self, message: str, code: int = 4001):
        self.message = message
        self.code = code
        super().__init__(self.message)


class WebSocketConnection(BaseModel):
    """Authenticated WebSocket connection metadata"""
    user_id: UUID
    username: str
    role: UserRole
    permissions: List[str]
    connected_at: datetime
    last_activity: datetime
    connection_type: str
    conversation_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class WebSocketAuthManager:
    """
    WebSocket Authentication Manager
    
    Provides secure authentication for WebSocket connections with:
    - JWT token validation
    - Connection authorization
    - Real-time token verification
    - Session management
    - Audit logging
    """
    
    def __init__(self):
        self.authenticated_connections: Dict[WebSocket, WebSocketConnection] = {}
        self.user_connections: Dict[str, List[WebSocket]] = {}
        self.connection_counts: Dict[str, int] = {}
        
        # Connection limits per user
        self.max_connections_per_user = 10
        
    async def authenticate_websocket(
        self,
        websocket: WebSocket,
        user_id: str,
        connection_type: str = "general",
        conversation_id: Optional[str] = None
    ) -> WebSocketConnection:
        """
        Authenticate WebSocket connection with JWT token validation
        
        Args:
            websocket: FastAPI WebSocket instance
            user_id: User ID from URL path parameter
            connection_type: Type of connection (general, conversation, web3)
            conversation_id: Optional conversation ID for conversation-specific connections
            
        Returns:
            WebSocketConnection: Authenticated connection metadata
            
        Raises:
            WebSocketAuthError: If authentication fails
        """
        try:
            # Extract authentication token from query parameters or headers
            token = await self._extract_auth_token(websocket)
            if not token:
                raise WebSocketAuthError("Authentication token required", 4001)
            
            # Validate JWT token
            token_data = await self._validate_token(token)
            if not token_data:
                raise WebSocketAuthError("Invalid or expired token", 4001)
            
            # Verify user ID matches token
            if str(token_data.user_id) != user_id:
                raise WebSocketAuthError("User ID mismatch", 4003)
            
            # Check connection limits
            if await self._check_connection_limits(user_id):
                raise WebSocketAuthError("Too many connections", 4008)
            
            # Verify conversation access if applicable
            if conversation_id:
                has_access = await self._verify_conversation_access(
                    token_data.user_id, conversation_id
                )
                if not has_access:
                    raise WebSocketAuthError("Access denied to conversation", 4003)
            
            # Get client info
            client_host = websocket.client.host if websocket.client else "unknown"
            user_agent = websocket.headers.get("user-agent", "unknown")
            
            # Create connection metadata
            connection = WebSocketConnection(
                user_id=token_data.user_id,
                username=token_data.username,
                role=UserRole(token_data.role),
                permissions=token_data.permissions,
                connected_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                connection_type=connection_type,
                conversation_id=conversation_id,
                ip_address=client_host,
                user_agent=user_agent
            )
            
            # Store connection
            self.authenticated_connections[websocket] = connection
            
            # Track user connections
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)
            
            # Update connection count
            self.connection_counts[user_id] = len(self.user_connections[user_id])
            
            # Log successful authentication
            await audit_logger.log_security_event(
                event_type="websocket_authenticated",
                user_id=str(token_data.user_id),
                details={
                    "connection_type": connection_type,
                    "conversation_id": conversation_id,
                    "ip_address": client_host,
                    "user_agent": user_agent
                },
                security_level="info"
            )
            
            logger.info("WebSocket authenticated successfully",
                       user_id=user_id,
                       connection_type=connection_type,
                       conversation_id=conversation_id,
                       ip_address=client_host)
            
            return connection
            
        except WebSocketAuthError:
            raise
        except Exception as e:
            logger.error("WebSocket authentication error", error=str(e))
            raise WebSocketAuthError("Authentication failed", 4001)
    
    async def disconnect_websocket(self, websocket: WebSocket):
        """
        Clean up disconnected WebSocket
        
        Args:
            websocket: WebSocket instance to disconnect
        """
        try:
            if websocket not in self.authenticated_connections:
                return
            
            connection = self.authenticated_connections[websocket]
            user_id = str(connection.user_id)
            
            # Remove from authenticated connections
            del self.authenticated_connections[websocket]
            
            # Remove from user connections
            if user_id in self.user_connections:
                self.user_connections[user_id] = [
                    ws for ws in self.user_connections[user_id] if ws != websocket
                ]
                
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
                    self.connection_counts.pop(user_id, None)
                else:
                    self.connection_counts[user_id] = len(self.user_connections[user_id])
            
            # Log disconnection
            await audit_logger.log_security_event(
                event_type="websocket_disconnected",
                user_id=user_id,
                details={
                    "connection_type": connection.connection_type,
                    "conversation_id": connection.conversation_id,
                    "duration_seconds": (
                        datetime.now(timezone.utc) - connection.connected_at
                    ).total_seconds()
                },
                security_level="info"
            )
            
            logger.info("WebSocket disconnected",
                       user_id=user_id,
                       connection_type=connection.connection_type)
            
        except Exception as e:
            logger.error("Error disconnecting WebSocket", error=str(e))
    
    async def verify_connection_auth(self, websocket: WebSocket) -> Optional[WebSocketConnection]:
        """
        Verify WebSocket connection is authenticated
        
        Args:
            websocket: WebSocket instance to verify
            
        Returns:
            WebSocketConnection if authenticated, None otherwise
        """
        connection = self.authenticated_connections.get(websocket)
        if not connection:
            return None
        
        # Update last activity
        connection.last_activity = datetime.now(timezone.utc)
        
        # Optionally verify token is still valid for long-lived connections
        # (implement token refresh logic here if needed)
        
        return connection
    
    async def require_permission(
        self,
        websocket: WebSocket,
        required_permission: str
    ) -> WebSocketConnection:
        """
        Require specific permission for WebSocket operation
        
        Args:
            websocket: WebSocket instance
            required_permission: Required permission
            
        Returns:
            WebSocketConnection if authorized
            
        Raises:
            WebSocketAuthError: If not authorized
        """
        connection = await self.verify_connection_auth(websocket)
        if not connection:
            raise WebSocketAuthError("Not authenticated", 4001)
        
        if required_permission not in connection.permissions:
            raise WebSocketAuthError(f"Permission required: {required_permission}", 4003)
        
        return connection
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        active_connections = len(self.authenticated_connections)
        unique_users = len(self.user_connections)
        
        connection_types = {}
        for connection in self.authenticated_connections.values():
            conn_type = connection.connection_type
            connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
        
        return {
            "active_connections": active_connections,
            "unique_users": unique_users,
            "connection_types": connection_types,
            "max_connections_per_user": self.max_connections_per_user
        }
    
    async def _extract_auth_token(self, websocket: WebSocket) -> Optional[str]:
        """Extract authentication token from WebSocket connection"""
        try:
            # Method 1: Check query parameters for token
            query_params = parse_qs(websocket.url.query)
            if "token" in query_params:
                return query_params["token"][0]
            
            # Method 2: Check Authorization header
            auth_header = websocket.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                return auth_header[7:]  # Remove "Bearer " prefix
            
            # Method 3: Check for token in cookies
            cookies = websocket.cookies
            if "access_token" in cookies:
                return cookies["access_token"]
            
            return None
            
        except Exception as e:
            logger.error("Error extracting auth token", error=str(e))
            return None
    
    async def _validate_token(self, token: str) -> Optional[Any]:
        """Validate JWT token"""
        try:
            # Use existing JWT handler to decode and validate token
            payload = jwt_handler.decode_token(token)
            if not payload:
                return None
            
            # Check if token is blacklisted
            if await jwt_handler.is_token_blacklisted(token):
                return None
            
            return payload
            
        except Exception as e:
            logger.error("Token validation error", error=str(e))
            return None
    
    async def _check_connection_limits(self, user_id: str) -> bool:
        """Check if user has exceeded connection limits"""
        current_connections = self.connection_counts.get(user_id, 0)
        return current_connections >= self.max_connections_per_user
    
    async def _verify_conversation_access(
        self,
        user_id: UUID,
        conversation_id: str
    ) -> bool:
        """
        Verify user has access to specific conversation
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID to check access for
            
        Returns:
            True if user has access, False otherwise
        """
        try:
            # Comprehensive conversation access control implementation
            db_service = get_database_service()
            if not db_service:
                logger.warning("Database service unavailable, denying conversation access",
                              user_id=str(user_id), conversation_id=conversation_id)
                return False  # Secure default: deny access if database unavailable
            
            # Get conversation from database
            conversation = await self._get_conversation_from_db(db_service, conversation_id)
            if not conversation:
                logger.warning("Conversation not found",
                              conversation_id=conversation_id,
                              user_id=str(user_id))
                return False
            
            # Check if user is the conversation owner
            if (conversation.get('owner_id') == str(user_id) or 
                conversation.get('user_id') == str(user_id)):
                logger.debug("Conversation access granted - owner",
                            user_id=str(user_id),
                            conversation_id=conversation_id)
                return True
            
            # Check for shared access permissions
            shared_access = await self._check_conversation_shared_access(db_service, conversation_id, str(user_id))
            if shared_access:
                logger.debug("Conversation access granted - shared access",
                            user_id=str(user_id),
                            conversation_id=conversation_id)
                return True
            
            # Check if user has admin privileges for system-wide access
            if await self._check_admin_conversation_access(str(user_id)):
                logger.debug("Conversation access granted - admin privileges",
                            user_id=str(user_id),
                            conversation_id=conversation_id)
                return True
            
            # Check if conversation is in a public workspace the user has access to
            workspace_access = await self._check_workspace_conversation_access(db_service, conversation_id, str(user_id))
            if workspace_access:
                logger.debug("Conversation access granted - workspace access",
                            user_id=str(user_id),
                            conversation_id=conversation_id)
                return True
            
            # Deny access by default
            logger.warning("Conversation access denied - insufficient permissions",
                          user_id=str(user_id),
                          conversation_id=conversation_id)
            return False
            
        except Exception as e:
            logger.error("Error verifying conversation access",
                        user_id=str(user_id),
                        conversation_id=conversation_id,
                        error=str(e))
            return False  # Secure default: deny access on error
    
    async def _get_conversation_from_db(self, db_service, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation from database with error handling"""
        try:
            # Use database service to fetch conversation
            # This would adapt to your specific database schema
            conversation = await db_service.get_record("conversations", {"id": conversation_id})
            return conversation
        except Exception as e:
            logger.error("Failed to fetch conversation from database",
                        conversation_id=conversation_id,
                        error=str(e))
            return None
    
    async def _check_conversation_shared_access(self, db_service, conversation_id: str, user_id: str) -> bool:
        """Check if user has shared access to conversation"""
        try:
            # Check conversation_permissions or conversation_shares table
            shared_access = await db_service.get_record(
                "conversation_permissions", 
                {"conversation_id": conversation_id, "user_id": user_id, "active": True}
            )
            return shared_access is not None
        except Exception as e:
            logger.error("Failed to check conversation shared access",
                        conversation_id=conversation_id,
                        user_id=user_id,
                        error=str(e))
            return False
    
    async def _check_admin_conversation_access(self, user_id: str) -> bool:
        """Check if user has admin privileges for conversation access"""
        try:
            # Import auth manager to check admin permissions
            from prsm.core.security.enhanced_authorization import get_enhanced_auth_manager
            # `from <submodule> import <fn>` resolves the real module (dodges the package
            # __init__ re-export that shadows the submodule name with the singleton instance).
            from prsm.core.auth.auth_manager import resolve_user_role

            # sp1264 — resolve the caller's REAL role (fail-safe to USER/deny). Hardcoding
            # UserRole.ADMIN here short-circuited check_permission to True, granting any
            # authenticated user system-wide conversation access (IDOR).
            real_role = await resolve_user_role(user_id)
            auth_manager = get_enhanced_auth_manager()
            has_admin_permission = await auth_manager.check_permission(
                user_id=user_id,
                user_role=real_role,
                resource_type="conversations",
                action="read_all"
            )
            return has_admin_permission
        except Exception as e:
            logger.error("Failed to check admin conversation access",
                        user_id=user_id,
                        error=str(e))
            return False
    
    async def _check_workspace_conversation_access(self, db_service, conversation_id: str, user_id: str) -> bool:
        """Check if conversation is in a workspace the user has access to"""
        try:
            # Check if conversation belongs to a workspace the user can access
            conversation_workspace = await db_service.get_record(
                "conversation_workspaces",
                {"conversation_id": conversation_id}
            )
            
            if not conversation_workspace:
                return False
            
            workspace_id = conversation_workspace.get("workspace_id")
            if not workspace_id:
                return False
            
            # Check user's workspace access
            workspace_access = await db_service.get_record(
                "workspace_members",
                {"workspace_id": workspace_id, "user_id": user_id, "active": True}
            )
            
            return workspace_access is not None
            
        except Exception as e:
            logger.error("Failed to check workspace conversation access",
                        conversation_id=conversation_id,
                        user_id=user_id,
                        error=str(e))
            return False


# Global WebSocket auth manager
websocket_auth = WebSocketAuthManager()


async def authenticate_websocket_connection(
    websocket: WebSocket,
    user_id: str,
    connection_type: str = "general",
    conversation_id: Optional[str] = None
) -> WebSocketConnection:
    """
    Helper function to authenticate WebSocket connections
    
    Usage in WebSocket endpoints:
        @app.websocket("/ws/{user_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            try:
                # Authenticate connection before accepting
                connection = await authenticate_websocket_connection(
                    websocket, user_id, "general"
                )
                await websocket.accept()
                # Connection is now authenticated...
            except WebSocketAuthError as e:
                await websocket.close(code=e.code, reason=e.message)
                return
    """
    return await websocket_auth.authenticate_websocket(
        websocket, user_id, connection_type, conversation_id
    )


async def require_websocket_permission(
    websocket: WebSocket,
    permission: str
) -> WebSocketConnection:
    """Helper function to require specific permission for WebSocket operations"""
    return await websocket_auth.require_permission(websocket, permission)


async def cleanup_websocket_connection(websocket: WebSocket):
    """Helper function to clean up WebSocket connections"""
    await websocket_auth.disconnect_websocket(websocket)