"""
Enhanced Authentication Middleware for PRSM API
Addresses audit team recommendations for comprehensive API authentication
"""

import time
import jwt
import hashlib
from typing import Dict, Any, Optional, Callable, Set
from datetime import datetime, timezone, timedelta
import structlog
from enum import Enum

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import (
    HTTP_401_UNAUTHORIZED, 
    HTTP_403_FORBIDDEN, 
    HTTP_429_TOO_MANY_REQUESTS
)

from prsm.core.config import get_settings
from prsm.core.redis_client import get_redis_client, resolve_async_redis
from prsm.core.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)
settings = get_settings()


class AuthenticationLevel(Enum):
    """Authentication security levels"""
    PUBLIC = "public"           # No authentication required
    BASIC = "basic"            # API key or basic token
    AUTHENTICATED = "authenticated"  # Valid JWT token
    VERIFIED = "verified"      # Verified user account
    PRIVILEGED = "privileged"  # Admin or system access
    SYSTEM = "system"          # Internal system calls


class Permission(Enum):
    """Granular permission system"""
    # Model permissions
    MODEL_READ = "model:read"
    MODEL_CREATE = "model:create"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_EXECUTE = "model:execute"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_CREATE = "data:create"
    DATA_UPDATE = "data:update"
    DATA_DELETE = "data:delete"
    
    # Token permissions
    TOKEN_VIEW = "token:view"
    TOKEN_TRANSFER = "token:transfer"
    TOKEN_MINT = "token:mint"
    TOKEN_BURN = "token:burn"
    
    # Agent permissions
    AGENT_CREATE = "agent:create"
    AGENT_EXECUTE = "agent:execute"
    AGENT_MANAGE = "agent:manage"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"


class RolePermissions:
    """Role-based permission mapping"""
    
    ROLE_PERMISSIONS = {
        "anonymous": set(),
        
        "user": {
            Permission.MODEL_READ,
            Permission.MODEL_EXECUTE,
            Permission.DATA_READ,
            Permission.TOKEN_VIEW,
            Permission.AGENT_EXECUTE,
        },
        
        "developer": {
            Permission.MODEL_READ,
            Permission.MODEL_CREATE,
            Permission.MODEL_EXECUTE,
            Permission.DATA_READ,
            Permission.DATA_CREATE,
            Permission.TOKEN_VIEW,
            Permission.TOKEN_TRANSFER,
            Permission.AGENT_CREATE,
            Permission.AGENT_EXECUTE,
        },
        
        "researcher": {
            Permission.MODEL_READ,
            Permission.MODEL_CREATE,
            Permission.MODEL_EXECUTE,
            Permission.DATA_READ,
            Permission.DATA_CREATE,
            Permission.DATA_UPDATE,
            Permission.TOKEN_VIEW,
            Permission.TOKEN_TRANSFER,
            Permission.AGENT_CREATE,
            Permission.AGENT_EXECUTE,
            Permission.AGENT_MANAGE,
        },
        
        "admin": {
            Permission.MODEL_READ,
            Permission.MODEL_CREATE,
            Permission.MODEL_UPDATE,
            Permission.MODEL_DELETE,
            Permission.MODEL_EXECUTE,
            Permission.DATA_READ,
            Permission.DATA_CREATE,
            Permission.DATA_UPDATE,
            Permission.DATA_DELETE,
            Permission.TOKEN_VIEW,
            Permission.TOKEN_TRANSFER,
            Permission.TOKEN_MINT,
            Permission.AGENT_CREATE,
            Permission.AGENT_EXECUTE,
            Permission.AGENT_MANAGE,
            Permission.SYSTEM_MONITOR,
        },
        
        "system": {perm for perm in Permission},  # All permissions
    }
    
    @classmethod
    def get_permissions(cls, role: str) -> Set[Permission]:
        """Get permissions for a role"""
        return cls.ROLE_PERMISSIONS.get(role, set())
    
    @classmethod
    def has_permission(cls, role: str, permission: Permission) -> bool:
        """Check if role has specific permission"""
        return permission in cls.get_permissions(role)


class EnhancedAuthMiddleware(BaseHTTPMiddleware):
    """
    Enhanced authentication middleware with comprehensive security features
    
    Features:
    - JWT token validation
    - Role-based access control (RBAC)
    - Granular permission system
    - Advanced rate limiting
    - Request fingerprinting
    - Anomaly detection
    - Audit logging
    - API key management
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        
        # Configuration
        self.jwt_secret = settings.jwt_secret_key or "dev-secret-change-in-production"
        self.jwt_algorithm = "HS256"
        self.token_expiry = timedelta(hours=24)
        
        # Rate limiting configuration
        self.rate_limits = {
            "anonymous": {"requests": 10, "window": 60},      # 10 req/min
            "user": {"requests": 100, "window": 60},          # 100 req/min
            "developer": {"requests": 500, "window": 60},     # 500 req/min
            "researcher": {"requests": 1000, "window": 60},   # 1000 req/min
            "admin": {"requests": 2000, "window": 60},        # 2000 req/min
            "system": {"requests": 10000, "window": 60},      # 10000 req/min
        }
        
        # Endpoint configuration
        self.endpoint_config = self._load_endpoint_config()
        
        # Security components
        self.redis_client = None
        self.bearer = HTTPBearer(auto_error=False)
        
    def _load_endpoint_config(self) -> Dict[str, Dict[str, Any]]:
        """Load endpoint authentication and permission requirements"""
        return {
            # Public endpoints - no auth required
            "/": {"auth_level": AuthenticationLevel.PUBLIC},
            "/health": {"auth_level": AuthenticationLevel.PUBLIC},
            "/docs": {"auth_level": AuthenticationLevel.PUBLIC},
            "/redoc": {"auth_level": AuthenticationLevel.PUBLIC},
            "/openapi.json": {"auth_level": AuthenticationLevel.PUBLIC},
            
            # Authentication endpoints
            "/auth/login": {"auth_level": AuthenticationLevel.PUBLIC},
            "/auth/register": {"auth_level": AuthenticationLevel.PUBLIC},
            "/auth/refresh": {"auth_level": AuthenticationLevel.BASIC},
            "/auth/logout": {"auth_level": AuthenticationLevel.AUTHENTICATED},
            
            # Model endpoints
            "/api/v1/models": {
                "GET": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.MODEL_READ]},
                "POST": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.MODEL_CREATE]},
            },
            "/api/v1/models/{model_id}": {
                "GET": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.MODEL_READ]},
                "PUT": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.MODEL_UPDATE]},
                "DELETE": {"auth_level": AuthenticationLevel.PRIVILEGED, "permissions": [Permission.MODEL_DELETE]},
            },
            "/api/v1/models/{model_id}/execute": {
                "POST": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.MODEL_EXECUTE]},
            },
            
            # Data endpoints
            "/api/v1/data": {
                "GET": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.DATA_READ]},
                "POST": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.DATA_CREATE]},
            },
            "/api/v1/data/{data_id}": {
                "GET": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.DATA_READ]},
                "PUT": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.DATA_UPDATE]},
                "DELETE": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.DATA_DELETE]},
            },
            
            # Token endpoints
            "/api/v1/tokens": {
                "GET": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.TOKEN_VIEW]},
            },
            "/api/v1/tokens/transfer": {
                "POST": {"auth_level": AuthenticationLevel.VERIFIED, "permissions": [Permission.TOKEN_TRANSFER]},
            },
            
            # Agent endpoints
            "/api/v1/agents": {
                "GET": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.AGENT_EXECUTE]},
                "POST": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.AGENT_CREATE]},
            },
            "/api/v1/agents/{agent_id}/execute": {
                "POST": {"auth_level": AuthenticationLevel.AUTHENTICATED, "permissions": [Permission.AGENT_EXECUTE]},
            },
            
            # Admin endpoints
            "/api/v1/admin": {
                "auth_level": AuthenticationLevel.PRIVILEGED,
                "permissions": [Permission.SYSTEM_ADMIN]
            },
            "/api/v1/system": {
                "auth_level": AuthenticationLevel.SYSTEM,
                "permissions": [Permission.SYSTEM_ADMIN]
            },
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method"""
        start_time = time.time()
        
        # Initialize Redis client
        if not self.redis_client:
            try:
                self.redis_client = get_redis_client()
            except Exception as e:
                logger.warning("Redis not available for auth middleware", error=str(e))
        
        # Extract request context
        context = await self._extract_request_context(request)
        
        try:
            # 1. Determine endpoint requirements
            endpoint_config = self._get_endpoint_config(request.url.path, request.method)
            
            # 2. Authenticate user
            auth_result = await self._authenticate_request(request, endpoint_config)
            if isinstance(auth_result, Response):
                return auth_result  # Authentication failed
            
            user_info, auth_level = auth_result
            
            # 3. Check authorization (permissions)
            auth_check = await self._check_authorization(user_info, endpoint_config, context)
            if isinstance(auth_check, Response):
                return auth_check  # Authorization failed
            
            # 4. Check rate limiting
            rate_limit_check = await self._check_enhanced_rate_limit(user_info, context)
            if isinstance(rate_limit_check, Response):
                return rate_limit_check  # Rate limit exceeded
            
            # 5. Detect anomalies
            await self._detect_anomalies(user_info, context)
            
            # 6. Add user context to request
            request.state.user = user_info
            request.state.auth_level = auth_level
            request.state.permissions = RolePermissions.get_permissions(user_info.get("role", "anonymous"))
            
            # 7. Process request
            response = await call_next(request)
            
            # 8. Log successful request
            processing_time = time.time() - start_time
            await self._log_successful_request(context, user_info, response.status_code, processing_time)
            
            # 9. Add security headers
            self._add_security_headers(response, user_info)
            
            return response
            
        except Exception as e:
            # Handle and log errors
            processing_time = time.time() - start_time
            return await self._handle_error(e, context, processing_time)
    
    async def _extract_request_context(self, request: Request) -> Dict[str, Any]:
        """Extract comprehensive request context for security analysis"""
        # Get client IP (handle proxy headers)
        client_ip = self._get_real_client_ip(request)
        
        # Create request fingerprint
        fingerprint = self._create_request_fingerprint(request)
        
        return {
            "ip": client_ip,
            "user_agent": request.headers.get("user-agent", ""),
            "path": request.url.path,
            "method": request.method,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "fingerprint": fingerprint,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request.headers.get("x-request-id", f"req_{int(time.time())}_{hash(str(request.url))%10000:04d}")
        }
    
    def _get_real_client_ip(self, request: Request) -> str:
        """Get real client IP, handling proxies and CDNs"""
        # Check various proxy headers in order of preference
        proxy_headers = [
            "cf-connecting-ip",      # Cloudflare
            "x-real-ip",            # Nginx
            "x-forwarded-for",      # Standard proxy
            "x-client-ip",          # Apache
            "x-cluster-client-ip",  # GCP Load Balancer
        ]
        
        for header in proxy_headers:
            if header in request.headers:
                ip = request.headers[header].split(",")[0].strip()
                if ip and ip != "unknown":
                    return ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"
    
    def _create_request_fingerprint(self, request: Request) -> str:
        """Create unique fingerprint for request pattern analysis"""
        fingerprint_data = {
            "user_agent": request.headers.get("user-agent", ""),
            "accept": request.headers.get("accept", ""),
            "accept_language": request.headers.get("accept-language", ""),
            "accept_encoding": request.headers.get("accept-encoding", ""),
            "connection": request.headers.get("connection", ""),
        }
        
        # Create hash of fingerprint data
        fingerprint_str = "|".join(f"{k}:{v}" for k, v in sorted(fingerprint_data.items()))
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    def _get_endpoint_config(self, path: str, method: str) -> Dict[str, Any]:
        """Get authentication configuration for endpoint"""
        # Try exact path match first
        if path in self.endpoint_config:
            config = self.endpoint_config[path]
            if isinstance(config, dict) and method in config:
                return config[method]
            elif "auth_level" in config:
                return config
        
        # Try pattern matching for parameterized paths
        for pattern, config in self.endpoint_config.items():
            if self._path_matches_pattern(path, pattern):
                if isinstance(config, dict) and method in config:
                    return config[method]
                elif "auth_level" in config:
                    return config
        
        # Default to authenticated for API endpoints
        if path.startswith("/api/"):
            return {
                "auth_level": AuthenticationLevel.AUTHENTICATED,
                "permissions": []
            }
        
        # Default to public for other endpoints
        return {"auth_level": AuthenticationLevel.PUBLIC}
    
    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern with parameters"""
        if "{" not in pattern:
            return path == pattern
        
        # Simple pattern matching for paths like /api/v1/models/{model_id}
        path_parts = path.split("/")
        pattern_parts = pattern.split("/")
        
        if len(path_parts) != len(pattern_parts):
            return False
        
        for path_part, pattern_part in zip(path_parts, pattern_parts):
            if pattern_part.startswith("{") and pattern_part.endswith("}"):
                continue  # Parameter match
            elif path_part != pattern_part:
                return False
        
        return True
    
    async def _authenticate_request(self, request: Request, endpoint_config: Dict[str, Any]) -> tuple:
        """Authenticate request based on endpoint requirements"""
        auth_level = endpoint_config.get("auth_level", AuthenticationLevel.PUBLIC)
        
        # Public endpoints don't need authentication
        if auth_level == AuthenticationLevel.PUBLIC:
            return {"role": "anonymous", "user_id": None}, auth_level
        
        # Extract authentication credentials
        auth_header = request.headers.get("authorization")
        api_key = request.headers.get("x-api-key")
        
        # Try JWT token authentication
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            user_info = await self._validate_jwt_token(token)
            if user_info:
                return user_info, AuthenticationLevel.AUTHENTICATED
        
        # Try API key authentication
        if api_key:
            user_info = await self._validate_api_key(api_key)
            if user_info:
                return user_info, AuthenticationLevel.BASIC
        
        # Authentication required but not provided
        if auth_level != AuthenticationLevel.PUBLIC:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return {"role": "anonymous", "user_id": None}, AuthenticationLevel.PUBLIC
    
    async def _validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user information"""
        try:
            # Decode and validate token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
                logger.warning("Expired JWT token")
                return None
            
            # Check if token is blacklisted (if Redis available)
            # sp1247 — resolve the inner async Redis (the wrapper has no .get).
            _redis = resolve_async_redis(self.redis_client)
            if _redis is not None:
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                if await _redis.get(f"blacklist:{token_hash}"):
                    logger.warning("Blacklisted JWT token")
                    return None
            
            # Return user information
            return {
                "user_id": payload.get("sub"),
                "email": payload.get("email"),
                "role": payload.get("role", "user"),
                "verified": payload.get("verified", False),
                "token_type": "jwt"
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", error=str(e))
            return None
        except Exception as e:
            logger.error("JWT validation error", error=str(e))
            return None
    
    async def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user information"""
        try:
            # In production, this would query a database or cache
            # For now, implement basic validation
            
            if not api_key or len(api_key) < 20:
                return None
            
            # Check if API key is blacklisted
            # sp1247 — resolve the inner async Redis (the wrapper has no .get).
            _redis = resolve_async_redis(self.redis_client)
            if _redis is not None:
                if await _redis.get(f"blacklist_key:{api_key}"):
                    logger.warning("Blacklisted API key")
                    return None
            
            # Mock API key validation (replace with actual implementation)
            # In production, query database for API key details
            api_key_prefix = api_key[:8]
            
            # Return basic user info for valid API keys
            return {
                "user_id": f"api_user_{api_key_prefix}",
                "role": "user",  # Default role for API key users
                "verified": False,
                "token_type": "api_key",
                "api_key_prefix": api_key_prefix
            }
            
        except Exception as e:
            logger.error("API key validation error", error=str(e))
            return None
    
    async def _check_authorization(self, user_info: Dict[str, Any], endpoint_config: Dict[str, Any], context: Dict[str, Any]) -> Optional[Response]:
        """Check if user has required permissions for endpoint"""
        required_permissions = endpoint_config.get("permissions", [])
        if not required_permissions:
            return None  # No specific permissions required
        
        user_role = user_info.get("role", "anonymous")
        user_permissions = RolePermissions.get_permissions(user_role)
        
        # Check if user has all required permissions
        missing_permissions = []
        for permission in required_permissions:
            if permission not in user_permissions:
                missing_permissions.append(permission.value)
        
        if missing_permissions:
            # Log authorization failure
            await audit_logger.log_security_event(
                "authorization_failure",
                {
                    "user_id": user_info.get("user_id"),
                    "role": user_role,
                    "required_permissions": [p.value for p in required_permissions],
                    "missing_permissions": missing_permissions
                },
                context
            )
            
            logger.warning("Authorization failed - insufficient permissions",
                          user_id=user_info.get("user_id"),
                          role=user_role,
                          missing_permissions=missing_permissions)
            
            return JSONResponse(
                status_code=HTTP_403_FORBIDDEN,
                content={
                    "detail": "Insufficient permissions",
                    "required_permissions": missing_permissions
                }
            )
        
        return None  # Authorization successful
    
    async def _check_enhanced_rate_limit(self, user_info: Dict[str, Any], context: Dict[str, Any]) -> Optional[Response]:
        """Enhanced rate limiting with user-based limits"""
        # sp1247 — resolve the live inner async Redis (self.redis_client is the
        # RedisClient wrapper, which has no .pipeline; calling it raised AttributeError
        # → the except failed OPEN). None → no live Redis → skip.
        _redis = resolve_async_redis(self.redis_client)
        if _redis is None:
            return None  # Skip if Redis not available

        user_role = user_info.get("role", "anonymous")
        user_id = user_info.get("user_id", context["ip"])
        
        # Get rate limit configuration for user role
        rate_config = self.rate_limits.get(user_role, self.rate_limits["anonymous"])
        
        try:
            # Create rate limit key
            rate_key = f"rate_limit:{user_role}:{user_id}"
            current_time = int(time.time())
            window_start = current_time - (current_time % rate_config["window"])
            window_key = f"{rate_key}:{window_start}"
            
            # Use Redis pipeline for atomic operations
            pipe = _redis.pipeline()
            pipe.incr(window_key)
            pipe.expire(window_key, rate_config["window"] * 2)
            results = await pipe.execute()
            
            current_requests = results[0]
            
            # Check if limit exceeded
            if current_requests > rate_config["requests"]:
                # Log rate limit violation
                await audit_logger.log_security_event(
                    "rate_limit_exceeded",
                    {
                        "user_id": user_id,
                        "role": user_role,
                        "requests": current_requests,
                        "limit": rate_config["requests"],
                        "window": rate_config["window"]
                    },
                    context
                )
                
                logger.warning("Rate limit exceeded",
                              user_id=user_id,
                              role=user_role,
                              requests=current_requests,
                              limit=rate_config["requests"])
                
                return JSONResponse(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after": rate_config["window"],
                        "limit": rate_config["requests"],
                        "window": rate_config["window"]
                    },
                    headers={"Retry-After": str(rate_config["window"])}
                )
            
            return None  # Rate limit OK
            
        except Exception as e:
            logger.error("Rate limit check error", error=str(e))
            return None  # Allow request on error
    
    async def _detect_anomalies(self, user_info: Dict[str, Any], context: Dict[str, Any]):
        """Detect suspicious request patterns"""
        # sp1247 — resolve the live inner async Redis (the wrapper has no .incr/.get;
        # calling them raised AttributeError → anomaly detection silently no-op'd).
        _redis = resolve_async_redis(self.redis_client)
        if _redis is None:
            return

        try:
            user_id = user_info.get("user_id", context["ip"])
            fingerprint = context["fingerprint"]

            # Track request patterns
            pattern_key = f"pattern:{user_id}:{fingerprint}"
            await _redis.incr(pattern_key)
            await _redis.expire(pattern_key, 3600)  # 1 hour window

            pattern_count = await _redis.get(pattern_key)
            pattern_count = int(pattern_count) if pattern_count else 0
            
            # Detect anomalies
            anomalies = []
            
            # High request frequency from same fingerprint
            if pattern_count > 100:  # More than 100 requests per hour
                anomalies.append("high_frequency_requests")
            
            # Suspicious user agent patterns
            user_agent = context["user_agent"].lower()
            if any(bot in user_agent for bot in ["bot", "crawler", "spider", "scraper"]):
                anomalies.append("bot_user_agent")
            
            # Empty or suspicious user agent
            if not user_agent or len(user_agent) < 10:
                anomalies.append("suspicious_user_agent")
            
            # Log anomalies
            if anomalies:
                await audit_logger.log_security_event(
                    "anomaly_detected",
                    {
                        "user_id": user_id,
                        "anomalies": anomalies,
                        "pattern_count": pattern_count,
                        "fingerprint": fingerprint
                    },
                    context
                )
                
                logger.warning("Request anomalies detected",
                              user_id=user_id,
                              anomalies=anomalies,
                              pattern_count=pattern_count)
        
        except Exception as e:
            logger.error("Anomaly detection error", error=str(e))
    
    async def _log_successful_request(self, context: Dict[str, Any], user_info: Dict[str, Any], 
                                    status_code: int, processing_time: float):
        """Log successful authenticated request"""
        try:
            log_data = {
                **context,
                "user_id": user_info.get("user_id"),
                "role": user_info.get("role"),
                "status_code": status_code,
                "processing_time": round(processing_time, 3)
            }
            
            # Log with appropriate level
            if status_code >= 400:
                logger.warning("Request completed with error", **log_data)
            else:
                logger.debug("Request completed successfully", **log_data)
            
            # Audit log
            await audit_logger.log_access_event(
                context["method"],
                context["path"],
                status_code,
                {**context, "user_info": user_info}
            )
            
        except Exception as e:
            logger.error("Request logging error", error=str(e))
    
    def _add_security_headers(self, response: Response, user_info: Dict[str, Any]):
        """Add security headers based on user context"""
        # Standard security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        # CSP based on user role
        user_role = user_info.get("role", "anonymous")
        if user_role in ["admin", "system"]:
            # More permissive CSP for admin interfaces
            security_headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' wss: https:"
            )
        else:
            # Strict CSP for regular users
            security_headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'"
            )
        
        # Apply headers
        for header, value in security_headers.items():
            response.headers[header] = value
    
    async def _handle_error(self, error: Exception, context: Dict[str, Any], processing_time: float) -> Response:
        """Handle and log middleware errors"""
        # Log error
        logger.error("Authentication middleware error",
                    error=str(error),
                    error_type=type(error).__name__,
                    processing_time=processing_time,
                    **context)
        
        # Audit log
        await audit_logger.log_security_event(
            "auth_middleware_error",
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "processing_time": processing_time
            },
            context
        )
        
        # Return appropriate error response
        if isinstance(error, HTTPException):
            return JSONResponse(
                status_code=error.status_code,
                content={"detail": error.detail}
            )
        else:
            # Generic error response to avoid information leakage
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
            
            # Add security headers even for errors
            self._add_security_headers(response, {"role": "anonymous"})
            return response


# Utility functions for token management

async def create_jwt_token(user_id: str, email: str, role: str, verified: bool = False) -> str:
    """Create JWT token for user"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "verified": verified,
        "iat": now.timestamp(),
        "exp": (now + timedelta(hours=24)).timestamp(),
        "iss": "prsm-api",
        "aud": "prsm-users"
    }
    
    secret = settings.jwt_secret_key or "dev-secret-change-in-production"
    return jwt.encode(payload, secret, algorithm="HS256")


async def blacklist_token(token: str, redis_client = None):
    """Blacklist a JWT token"""
    if not redis_client:
        redis_client = get_redis_client()

    # sp1247 — resolve the live inner async Redis (the wrapper has no .setex; the
    # blacklist WRITE was silently failing, so a revoked token was never recorded).
    _redis = resolve_async_redis(redis_client)
    if _redis is not None:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        # Blacklist for 24 hours (token expiry)
        await _redis.setex(f"blacklist:{token_hash}", 86400, "blacklisted")


async def create_api_key(user_id: str, description: str = "") -> str:
    """Create API key for user"""
    import secrets
    
    # Generate secure random API key
    api_key = f"prsm_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"
    
    # In production, store API key details in database
    # For now, just return the key
    return api_key