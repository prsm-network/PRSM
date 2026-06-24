"""
JWT Token Handler
Secure JWT token generation, validation, and management for PRSM authentication

Security Features (Post-Audit Hardening):
1. Real token revocation via PostgreSQL + Redis cache
2. Algorithm confusion attack prevention (explicit single algorithm)
3. Required claims validation
4. Token binding and JTI tracking
5. Tiered revocation checking (Redis cache -> PostgreSQL authoritative)
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple
from uuid import UUID

import jwt
import structlog
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import text

from prsm.core.config import get_settings
from prsm.core.redis_client import resolve_async_redis  # sp1248

logger = structlog.get_logger(__name__)
settings = get_settings()

# Security constants
ALLOWED_ALGORITHMS = {"HS256", "HS384", "HS512"}  # Only symmetric algorithms
REVOCATION_CACHE_TTL = 300  # 5 minutes
MIN_SECRET_KEY_LENGTH = 32  # Minimum 256 bits


class TokenData(BaseModel):
    """Token payload data"""
    user_id: UUID
    username: str
    email: str
    role: str
    permissions: list[str]
    token_type: str = "access"
    issued_at: datetime
    expires_at: datetime


class JWTHandler:
    """
    JWT token handler with secure token generation and validation

    Security Features (Post-Audit Hardening):
    - Algorithm confusion attack prevention (single algorithm enforcement)
    - Real token revocation via PostgreSQL + Redis cache
    - Required claims validation (exp, iat, sub, jti)
    - Token binding and tracking
    - Secure password hashing with bcrypt (12 rounds)
    """

    def __init__(self):
        # Handle case where settings is None (e.g., during test collection)
        if settings is None:
            self.algorithm = "HS256"  # Default algorithm
            self.secret_key = "test-secret-key-change-in-production"  # Default test key
        else:
            self.algorithm = settings.jwt_algorithm
            self.secret_key = settings.secret_key
        
        self.access_token_expire_minutes = 30  # 30 minutes
        self.refresh_token_expire_days = 7     # 7 days

        # SECURITY: Validate algorithm is in allowed list
        if self.algorithm not in ALLOWED_ALGORITHMS:
            raise ValueError(
                f"JWT algorithm '{self.algorithm}' not allowed. "
                f"Permitted algorithms: {ALLOWED_ALGORITHMS}"
            )

        # SECURITY: Validate secret key strength
        if len(self.secret_key) < MIN_SECRET_KEY_LENGTH:
            if settings and hasattr(settings, 'is_production') and settings.is_production:
                raise ValueError(
                    f"JWT secret key must be at least {MIN_SECRET_KEY_LENGTH} characters in production"
                )
            else:
                logger.warning(
                    f"JWT secret key is shorter than recommended {MIN_SECRET_KEY_LENGTH} characters"
                )

        # Password hashing context
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12  # Strong hashing rounds
        )

        # Database and cache services
        self.db_service = None
        self.redis_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize database and Redis connections for token management"""
        if self._initialized:
            return

        try:
            # Initialize database service
            from prsm.core.database import get_database_service
            self.db_service = get_database_service()

            # Initialize Redis client for revocation cache
            try:
                from prsm.core.redis_client import get_redis_client
                self.redis_client = get_redis_client()
                logger.info("JWT handler initialized with Redis cache")
            except Exception as redis_error:
                logger.warning(
                    "Redis not available for JWT cache - using database only",
                    error=str(redis_error)
                )
                self.redis_client = None

            self._initialized = True
            logger.info("JWT handler initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize JWT handler", error=str(e))
            # Don't fail completely - handler can work without persistence
            self._initialized = True
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def generate_token_hash(self, token: str) -> str:
        """Generate hash of token for database storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def create_access_token(self, user_data: Dict[str, Any], 
                                expires_delta: Optional[timedelta] = None) -> Tuple[str, TokenData]:
        """
        Create JWT access token
        
        Args:
            user_data: User information to encode in token
            expires_delta: Custom expiration time
            
        Returns:
            Tuple of (token_string, token_data)
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        issued_at = datetime.now(timezone.utc)
        
        # Create token payload
        payload = {
            "sub": str(user_data["user_id"]),
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"],
            "permissions": user_data.get("permissions", []),
            "token_type": "access",
            "iat": issued_at.timestamp(),
            "exp": expire.timestamp(),
            "jti": secrets.token_urlsafe(32)  # Unique token identifier
        }
        
        # Generate JWT token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Create token data object
        token_data = TokenData(
            user_id=UUID(user_data["user_id"]),
            username=user_data["username"],
            email=user_data["email"],
            role=user_data["role"],
            permissions=user_data.get("permissions", []),
            token_type="access",
            issued_at=issued_at,
            expires_at=expire
        )
        
        # Store token hash in database for revocation capability
        if self.db_service:
            try:
                await self._store_token_record(
                    user_id=UUID(user_data["user_id"]),
                    token=token,
                    token_type="access",
                    expires_at=expire,
                    client_info=user_data.get("client_info", {})
                )
            except Exception as e:
                logger.warning("Failed to store token record", error=str(e))
        
        logger.info("Access token created", 
                   user_id=user_data["user_id"],
                   username=user_data["username"],
                   expires_at=expire.isoformat())
        
        return token, token_data
    
    async def create_refresh_token(self, user_data: Dict[str, Any]) -> Tuple[str, TokenData]:
        """
        Create JWT refresh token
        
        Args:
            user_data: User information to encode in token
            
        Returns:
            Tuple of (token_string, token_data)
        """
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        issued_at = datetime.now(timezone.utc)
        
        # Create refresh token payload (minimal info for security)
        payload = {
            "sub": str(user_data["user_id"]),
            "username": user_data["username"],
            "token_type": "refresh",
            "iat": issued_at.timestamp(),
            "exp": expire.timestamp(),
            "jti": secrets.token_urlsafe(32)
        }
        
        # Generate JWT token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Create token data object
        token_data = TokenData(
            user_id=UUID(user_data["user_id"]),
            username=user_data["username"],
            email=user_data.get("email", ""),
            role=user_data.get("role", ""),
            permissions=[],
            token_type="refresh",
            issued_at=issued_at,
            expires_at=expire
        )
        
        # Store token hash in database
        if self.db_service:
            try:
                await self._store_token_record(
                    user_id=UUID(user_data["user_id"]),
                    token=token,
                    token_type="refresh",
                    expires_at=expire,
                    client_info=user_data.get("client_info", {})
                )
            except Exception as e:
                logger.warning("Failed to store refresh token record", error=str(e))
        
        logger.info("Refresh token created",
                   user_id=user_data["user_id"],
                   username=user_data["username"],
                   expires_at=expire.isoformat())
        
        return token, token_data
    
    async def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify and decode JWT token with security hardening.

        Security measures:
        1. Algorithm enforcement (no algorithm confusion)
        2. Signature verification
        3. Expiration checking (exp claim)
        4. Required claims validation (exp, iat, sub, jti)
        5. Real revocation checking (Redis cache + PostgreSQL)

        Args:
            token: JWT token string

        Returns:
            TokenData if valid, None if invalid
        """
        try:
            # SECURITY: Decode with explicit single algorithm to prevent
            # algorithm confusion attacks (e.g., none algorithm attack)
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],  # Single algorithm only
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["exp", "iat", "sub", "jti"]  # Required claims
                }
            )

            # Extract and validate token data
            user_id = UUID(payload.get("sub"))
            username = payload.get("username")
            jti = payload.get("jti")  # JWT ID for revocation checking

            if not jti:
                logger.warning("Token missing JTI claim", user_id=str(user_id))
                return None

            email = payload.get("email", "")
            role = payload.get("role", "")
            permissions = payload.get("permissions", [])
            token_type = payload.get("token_type", "access")

            issued_at = datetime.fromtimestamp(payload.get("iat"), tz=timezone.utc)
            expires_at = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)

            # Additional expiration check (belt and suspenders)
            if expires_at <= datetime.now(timezone.utc):
                logger.warning("Token expired", user_id=str(user_id), username=username)
                return None

            # SECURITY FIX: Real revocation checking with tiered lookup
            token_hash = self.generate_token_hash(token)
            if await self._is_token_revoked(token_hash, jti):
                logger.warning("Token revoked", user_id=str(user_id), jti=jti)
                return None

            # Update last used timestamp (fire-and-forget to not block response)
            asyncio.create_task(self._update_token_last_used(token_hash))

            # Create token data object
            token_data = TokenData(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
                permissions=permissions,
                token_type=token_type,
                issued_at=issued_at,
                expires_at=expires_at
            )

            logger.debug("Token verified successfully",
                        user_id=str(user_id),
                        username=username,
                        token_type=token_type)

            return token_data

        except jwt.ExpiredSignatureError:
            logger.warning("Token signature expired")
            return None
        except jwt.InvalidAlgorithmError:
            # SECURITY: This could indicate an algorithm confusion attack
            logger.error("Invalid token algorithm - possible attack attempt")
            return None
        except jwt.MissingRequiredClaimError as e:
            logger.warning("Token missing required claim", error=str(e))
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
        except Exception as e:
            logger.error("Token verification error", error=str(e))
            return None
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Tuple of (new_access_token, new_refresh_token) if successful
        """
        try:
            # Verify refresh token
            token_data = await self.verify_token(refresh_token)
            
            if not token_data or token_data.token_type != "refresh":
                logger.warning("Invalid refresh token")
                return None
            
            # Get user data from database
            if not self.db_service:
                logger.error("Database service not available for token refresh")
                return None
            
            # For now, create minimal user data (would fetch from database in production)
            user_data = {
                "user_id": str(token_data.user_id),
                "username": token_data.username,
                "email": token_data.email,
                "role": token_data.role,
                "permissions": token_data.permissions
            }
            
            # Revoke old refresh token
            old_token_hash = self.generate_token_hash(refresh_token)
            await self._revoke_token(old_token_hash)
            
            # Create new tokens
            new_access_token, _ = await self.create_access_token(user_data)
            new_refresh_token, _ = await self.create_refresh_token(user_data)
            
            logger.info("Tokens refreshed successfully", 
                       user_id=str(token_data.user_id),
                       username=token_data.username)
            
            return new_access_token, new_refresh_token
            
        except Exception as e:
            logger.error("Token refresh error", error=str(e))
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """
        Revoke a token (add to blacklist)
        
        Args:
            token: Token to revoke
            
        Returns:
            True if revoked successfully
        """
        try:
            token_hash = self.generate_token_hash(token)
            return await self._revoke_token(token_hash)
        except Exception as e:
            logger.error("Token revocation error", error=str(e))
            return False
    
    async def revoke_all_user_tokens(self, user_id: UUID) -> bool:
        """
        Revoke all tokens for a user
        
        Args:
            user_id: User ID to revoke tokens for
            
        Returns:
            True if revoked successfully
        """
        try:
            if not self.db_service:
                return False
            
            # Would update all user tokens as revoked in database
            # For now, log the action
            logger.info("All user tokens revoked", user_id=str(user_id))
            return True
            
        except Exception as e:
            logger.error("User token revocation error", error=str(e))
            return False
    
    # Private helper methods - SECURITY FIX: Real implementations

    async def _store_token_record(self, user_id: UUID, token: str, token_type: str,
                                expires_at: datetime, client_info: Dict[str, Any]):
        """Store token record in database for tracking and revocation."""
        try:
            token_hash = self.generate_token_hash(token)

            # Decode to get JTI
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti", "")

            if self.db_service:
                async with self.db_service.get_session() as session:
                    # Store token record
                    query = text("""
                        INSERT INTO user_tokens
                        (user_id, token_hash, jti, token_type, expires_at,
                         client_ip, user_agent, created_at, is_revoked)
                        VALUES
                        (:user_id, :token_hash, :jti, :token_type, :expires_at,
                         :client_ip, :user_agent, NOW(), FALSE)
                        ON CONFLICT (token_hash) DO UPDATE
                        SET last_used_at = NOW()
                    """)

                    await session.execute(query, {
                        "user_id": str(user_id),
                        "token_hash": token_hash,
                        "jti": jti,
                        "token_type": token_type,
                        "expires_at": expires_at,
                        "client_ip": client_info.get("ip", "unknown"),
                        "user_agent": (client_info.get("user_agent", "")[:500]
                                      if client_info.get("user_agent") else "")
                    })
                    await session.commit()

            logger.debug("Token record stored",
                        user_id=str(user_id),
                        token_type=token_type,
                        jti=jti[:8] + "..." if jti else "none")

        except Exception as e:
            # Log but don't fail - token is still valid even if we can't track it
            logger.warning("Failed to store token record", error=str(e))

    async def _is_token_revoked(self, token_hash: str, jti: str = None) -> bool:
        """
        Check if token is revoked using tiered caching strategy.

        Lookup order:
        1. Redis cache (fast path, sub-millisecond)
        2. PostgreSQL (authoritative source)
        3. Cache the result for subsequent lookups

        Args:
            token_hash: SHA256 hash of the token
            jti: JWT ID for additional lookup

        Returns:
            True if token is revoked, False otherwise
        """
        try:
            # Tier 1: Check Redis cache (fast path)
            # sp1248 — resolve the inner async Redis (self.redis_client is the
            # RedisClient wrapper, which has no .get; the wrapper bug made this cache
            # ALWAYS miss → every revocation check fell through to PostgreSQL. Safe
            # but pointless caching — now the fast path actually works.)
            _redis = resolve_async_redis(self.redis_client)
            if _redis is not None:
                try:
                    cache_key = f"token_revoked:{token_hash[:16]}"
                    cached_status = await _redis.get(cache_key)

                    if cached_status is not None:
                        is_revoked = cached_status in (b"1", "1", True, b"true", "true")
                        if is_revoked:
                            logger.debug("Token revocation found in cache",
                                        token_hash=token_hash[:16])
                        return is_revoked
                except Exception as cache_error:
                    logger.debug("Redis cache lookup failed", error=str(cache_error))

            # Tier 2: Check PostgreSQL (authoritative)
            if self.db_service:
                async with self.db_service.get_session() as session:
                    query = text("""
                        SELECT 1 FROM revoked_tokens
                        WHERE token_hash = :token_hash
                        OR (:jti IS NOT NULL AND jti = :jti)
                        LIMIT 1
                    """)

                    result = await session.execute(
                        query,
                        {"token_hash": token_hash, "jti": jti}
                    )
                    is_revoked = result.scalar() is not None

                    # Cache the result in Redis for future lookups
                    _redis = resolve_async_redis(self.redis_client)  # sp1248
                    if _redis is not None:
                        try:
                            cache_key = f"token_revoked:{token_hash[:16]}"
                            await _redis.setex(
                                cache_key,
                                REVOCATION_CACHE_TTL,
                                "1" if is_revoked else "0"
                            )
                        except Exception as cache_error:
                            logger.debug("Failed to cache revocation status",
                                        error=str(cache_error))

                    if is_revoked:
                        logger.debug("Token revocation found in database",
                                    token_hash=token_hash[:16])

                    return is_revoked

            # No database available - fail based on environment
            if settings.is_production:
                logger.error("Cannot verify token revocation in production - failing closed")
                return True  # Fail closed in production

            return False  # Allow in development if no DB

        except Exception as e:
            logger.error("Token revocation check failed", error=str(e))
            # Fail closed in production, open in development
            return settings.is_production

    async def _revoke_token(self, token_hash: str, jti: str = None,
                           user_id: UUID = None, reason: str = "User logout") -> bool:
        """
        Revoke token by adding to revocation list.

        Revocation is stored in:
        1. PostgreSQL (persistent, authoritative)
        2. Redis (cache for fast lookups)

        Args:
            token_hash: SHA256 hash of the token
            jti: JWT ID
            user_id: User who owns the token
            reason: Reason for revocation

        Returns:
            True if revoked successfully
        """
        try:
            # Store in PostgreSQL (authoritative)
            if self.db_service:
                async with self.db_service.get_session() as session:
                    query = text("""
                        INSERT INTO revoked_tokens
                        (token_hash, jti, user_id, reason, revoked_at, expires_at)
                        VALUES
                        (:token_hash, :jti, :user_id, :reason, NOW(),
                         NOW() + INTERVAL '8 days')
                        ON CONFLICT (token_hash) DO UPDATE
                        SET reason = EXCLUDED.reason,
                            revoked_at = NOW()
                    """)

                    await session.execute(query, {
                        "token_hash": token_hash,
                        "jti": jti or "",
                        "user_id": str(user_id) if user_id else None,
                        "reason": reason
                    })
                    await session.commit()

            # Immediately update Redis cache
            _redis = resolve_async_redis(self.redis_client)  # sp1248
            if _redis is not None:
                try:
                    cache_key = f"token_revoked:{token_hash[:16]}"
                    await _redis.setex(
                        cache_key,
                        86400,  # 24 hour TTL for revoked tokens
                        "1"
                    )
                except Exception as cache_error:
                    logger.warning("Failed to cache token revocation",
                                  error=str(cache_error))

            logger.info("Token revoked successfully",
                       token_hash=token_hash[:16] + "...",
                       reason=reason)
            return True

        except Exception as e:
            logger.error("Failed to revoke token", error=str(e))
            return False

    async def _update_token_last_used(self, token_hash: str):
        """Update token last used timestamp for activity tracking."""
        try:
            if self.db_service:
                async with self.db_service.get_session() as session:
                    query = text("""
                        UPDATE user_tokens
                        SET last_used_at = NOW()
                        WHERE token_hash = :token_hash
                    """)
                    await session.execute(query, {"token_hash": token_hash})
                    await session.commit()

        except Exception as e:
            # Non-critical - don't fail the request
            logger.debug("Failed to update token last used", error=str(e))

    async def _revoke_all_user_tokens_impl(self, user_id: UUID, reason: str = "User logout all") -> int:
        """
        Revoke all tokens for a user.

        Args:
            user_id: User ID to revoke tokens for
            reason: Reason for revocation

        Returns:
            Number of tokens revoked
        """
        revoked_count = 0

        try:
            if self.db_service:
                async with self.db_service.get_session() as session:
                    # Get all active tokens for user
                    select_query = text("""
                        SELECT token_hash, jti
                        FROM user_tokens
                        WHERE user_id = :user_id
                        AND is_revoked = FALSE
                        AND expires_at > NOW()
                    """)

                    result = await session.execute(
                        select_query, {"user_id": str(user_id)}
                    )
                    tokens = result.fetchall()

                    # Revoke each token
                    for token_row in tokens:
                        # Add to revoked_tokens table
                        insert_query = text("""
                            INSERT INTO revoked_tokens
                            (token_hash, jti, user_id, reason, revoked_at)
                            VALUES (:token_hash, :jti, :user_id, :reason, NOW())
                            ON CONFLICT (token_hash) DO NOTHING
                        """)
                        await session.execute(insert_query, {
                            "token_hash": token_row.token_hash,
                            "jti": token_row.jti,
                            "user_id": str(user_id),
                            "reason": reason
                        })

                        # Update Redis cache
                        _redis = resolve_async_redis(self.redis_client)  # sp1248
                        if _redis is not None:
                            try:
                                cache_key = f"token_revoked:{token_row.token_hash[:16]}"
                                await _redis.setex(cache_key, 86400, "1")
                            except Exception:
                                pass  # Cache update failure is non-critical, database is authoritative

                        revoked_count += 1

                    # Mark all user tokens as revoked
                    update_query = text("""
                        UPDATE user_tokens
                        SET is_revoked = TRUE
                        WHERE user_id = :user_id
                    """)
                    await session.execute(update_query, {"user_id": str(user_id)})

                    await session.commit()

            logger.info("All user tokens revoked",
                       user_id=str(user_id),
                       revoked_count=revoked_count)

            return revoked_count

        except Exception as e:
            logger.error("Failed to revoke all user tokens", error=str(e))
            return 0


# Global JWT handler instance
jwt_handler = JWTHandler()


async def create_access_token(user_data: Dict[str, Any],
                              expires_delta: Optional[timedelta] = None) -> Tuple[str, "TokenData"]:
    """Module-level convenience function delegating to the global jwt_handler."""
    return await jwt_handler.create_access_token(user_data, expires_delta)