"""
Web3 Frontend Integration for PRSM FTNS Token System

Provides API endpoints and WebSocket handlers for frontend Web3 integration
including wallet connection, transaction management, and real-time updates.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from prsm.core.config import get_settings
from .wallet_connector import Web3WalletConnector
from .contract_interface import FTNSContractInterface
from .faucet_integration import PolygonFaucetIntegration

logger = logging.getLogger(__name__)

# === Pydantic Models ===

class WalletConnectionRequest(BaseModel):
    network: str = Field(default="polygon_mumbai", description="Network to connect to")
    private_key: Optional[str] = Field(default=None, description="Wallet private key")
    mnemonic: Optional[str] = Field(default=None, description="Wallet mnemonic")

class TransferRequest(BaseModel):
    to_address: str = Field(description="Recipient wallet address")
    amount: Decimal = Field(description="Amount to transfer")
    gas_price: Optional[int] = Field(default=None, description="Gas price in wei")

class FaucetRequest(BaseModel):
    wallet_address: str = Field(description="Wallet address for MATIC request")
    preferred_faucet: Optional[str] = Field(default=None, description="Preferred faucet provider")

class WalletInfoResponse(BaseModel):
    address: str
    balance_matic: Decimal
    balance_ftns: Decimal
    network: str
    connected: bool
    detailed_balance: Optional[Dict] = None

class TransactionResponse(BaseModel):
    hash: str
    success: bool
    gas_used: int
    block_number: int
    error: Optional[str] = None
    timestamp: datetime

class Web3ConnectionManager:
    """Manages Web3 connections for multiple users"""
    
    def __init__(self):
        self.connections: Dict[str, Web3WalletConnector] = {}
        self.contracts: Dict[str, FTNSContractInterface] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
        
    async def get_user_connector(self, user_id: str) -> Optional[Web3WalletConnector]:
        """Get Web3 connector for user"""
        return self.connections.get(user_id)
        
    async def create_user_connector(self, user_id: str, config: Optional[Dict] = None) -> Web3WalletConnector:
        """Create new Web3 connector for user"""
        connector = Web3WalletConnector(config)
        self.connections[user_id] = connector
        return connector
        
    async def get_user_contract_interface(self, user_id: str) -> Optional[FTNSContractInterface]:
        """Get contract interface for user"""
        return self.contracts.get(user_id)
        
    async def create_user_contract_interface(self, user_id: str, 
                                           wallet_connector: Web3WalletConnector,
                                           contract_addresses: Dict[str, str]) -> FTNSContractInterface:
        """Create contract interface for user"""
        interface = FTNSContractInterface(wallet_connector, contract_addresses)
        await interface.initialize_contracts()
        self.contracts[user_id] = interface
        return interface
        
    async def add_websocket_connection(self, user_id: str, websocket: WebSocket):
        """Add WebSocket connection for user"""
        if user_id not in self.websocket_connections:
            self.websocket_connections[user_id] = []
        self.websocket_connections[user_id].append(websocket)
        
    async def remove_websocket_connection(self, user_id: str, websocket: WebSocket):
        """Remove WebSocket connection for user"""
        if user_id in self.websocket_connections:
            try:
                self.websocket_connections[user_id].remove(websocket)
                if not self.websocket_connections[user_id]:
                    del self.websocket_connections[user_id]
            except ValueError:
                pass
                
    async def broadcast_to_user(self, user_id: str, message: Dict):
        """Broadcast message to all user's WebSocket connections"""
        if user_id in self.websocket_connections:
            disconnected = []
            for websocket in self.websocket_connections[user_id]:
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                await self.remove_websocket_connection(user_id, ws)

# Global connection manager
connection_manager = Web3ConnectionManager()

# === API Router ===

router = APIRouter(prefix="/web3", tags=["Web3 Integration"])
security = HTTPBearer()
settings = get_settings()


async def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Get current user from JWT token via the CANONICAL verifier.

    sp1267 (audit round 5): the previous ad-hoc pyjwt.decode verified signature + exp but
    NEVER checked token_type or revocation — so a 7-day REFRESH token (same HS256 secret,
    carries sub+exp) worked as a full bearer credential on these wallet endpoints, and a
    logged-out / revoked token still authenticated. Routing through
    JWTHandler.verify_token enforces signature, expiry, required claims, REVOCATION, and lets
    us reject any non-access token.
    """
    from prsm.core.auth.jwt_handler import jwt_handler

    try:
        token_data = await jwt_handler.verify_token(token.credentials)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — never authenticate on a verifier error
        logger.error(f"JWT verification error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

    if token_data is None:
        # invalid signature / expired / malformed / REVOKED
        raise HTTPException(status_code=401, detail="Invalid or revoked token")
    if token_data.token_type != "access":
        # a refresh (or any non-access) token must NOT authenticate API requests
        raise HTTPException(status_code=401, detail="Invalid token type (expected access token)")

    return str(token_data.user_id)

@router.post("/connect")
async def connect_wallet(
    request: WalletConnectionRequest,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Connect Web3 wallet to specified network
    """
    try:
        # Get or create wallet connector
        connector = await connection_manager.get_user_connector(user_id)
        if not connector:
            connector = await connection_manager.create_user_connector(user_id)
        
        # Connect to network
        network_connected = await connector.connect_to_network(request.network)
        if not network_connected:
            raise HTTPException(status_code=400, detail=f"Failed to connect to network: {request.network}")
        
        # Connect wallet if credentials provided
        wallet_address = None
        if request.private_key or request.mnemonic:
            wallet_address = await connector.connect_wallet(
                private_key=request.private_key,
                mnemonic=request.mnemonic
            )
            if not wallet_address:
                raise HTTPException(status_code=400, detail="Failed to connect wallet")
        
        # Get wallet info
        wallet_info = await connector.get_wallet_info() if wallet_address else None
        
        # Broadcast connection status to user's WebSocket connections
        await connection_manager.broadcast_to_user(user_id, {
            "type": "wallet_connected",
            "network": request.network,
            "address": wallet_address,
            "wallet_info": wallet_info.__dict__ if wallet_info else None
        })
        
        return {
            "success": True,
            "network": request.network,
            "wallet_address": wallet_address,
            "wallet_info": wallet_info.__dict__ if wallet_info else None
        }
        
    except Exception as e:
        logger.error(f"Wallet connection failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallet/info")
async def get_wallet_info(
    user_id: str = Depends(get_current_user)
) -> WalletInfoResponse:
    """
    Get comprehensive wallet information
    """
    try:
        connector = await connection_manager.get_user_connector(user_id)
        if not connector:
            raise HTTPException(status_code=404, detail="Wallet not connected")
        
        wallet_info = await connector.get_wallet_info()
        if not wallet_info:
            raise HTTPException(status_code=404, detail="Wallet information not available")
        
        # Get detailed balance if contract interface is available
        detailed_balance = None
        contract_interface = await connection_manager.get_user_contract_interface(user_id)
        if contract_interface:
            balance_info = await contract_interface.get_detailed_balance()
            if balance_info:
                detailed_balance = {
                    "liquid": float(balance_info.liquid),
                    "locked": float(balance_info.locked),
                    "staked": float(balance_info.staked),
                    "total": float(balance_info.total),
                    "context_allocated": float(balance_info.context_allocated)
                }
        
        return WalletInfoResponse(
            address=wallet_info.address,
            balance_matic=wallet_info.balance_matic,
            balance_ftns=wallet_info.balance_ftns,
            network=wallet_info.network.value,
            connected=wallet_info.connected,
            detailed_balance=detailed_balance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get wallet info for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/contracts/initialize")
async def initialize_contracts(
    contract_addresses: Dict[str, str],
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Initialize smart contract interfaces
    """
    try:
        connector = await connection_manager.get_user_connector(user_id)
        if not connector or not connector.is_connected:
            raise HTTPException(status_code=400, detail="Wallet not connected")
        
        # Create contract interface
        interface = await connection_manager.create_user_contract_interface(
            user_id, connector, contract_addresses
        )
        
        # Get token info
        token_info = await interface.get_token_info()
        
        # Broadcast contract initialization to user's WebSocket connections
        await connection_manager.broadcast_to_user(user_id, {
            "type": "contracts_initialized",
            "token_info": token_info,
            "contract_addresses": contract_addresses
        })
        
        return {
            "success": True,
            "token_info": token_info,
            "contract_addresses": contract_addresses
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contract initialization failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transfer")
async def transfer_tokens(
    request: TransferRequest,
    user_id: str = Depends(get_current_user)
) -> TransactionResponse:
    """
    Transfer FTNS tokens to another address
    """
    try:
        interface = await connection_manager.get_user_contract_interface(user_id)
        if not interface:
            raise HTTPException(status_code=400, detail="Contract interface not initialized")
        
        # Execute transfer
        result = await interface.transfer_tokens(request.to_address, request.amount)
        if not result:
            raise HTTPException(status_code=500, detail="Transfer failed")
        
        # Broadcast transaction to user's WebSocket connections
        await connection_manager.broadcast_to_user(user_id, {
            "type": "transaction_sent",
            "transaction": {
                "hash": result.hash,
                "type": "transfer",
                "to": request.to_address,
                "amount": float(request.amount),
                "success": result.success
            }
        })
        
        return TransactionResponse(
            hash=result.hash,
            success=result.success,
            gas_used=result.gas_used,
            block_number=result.block_number,
            error=result.error,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token transfer failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/balance/{address}")
async def get_token_balance(
    address: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get FTNS token balance for specific address
    """
    try:
        interface = await connection_manager.get_user_contract_interface(user_id)
        if not interface:
            raise HTTPException(status_code=400, detail="Contract interface not initialized")
        
        # Get balance
        balance = await interface.get_balance(address)
        detailed_balance = await interface.get_detailed_balance(address)
        
        return {
            "address": address,
            "balance": float(balance),
            "detailed_balance": {
                "liquid": float(detailed_balance.liquid) if detailed_balance else 0,
                "locked": float(detailed_balance.locked) if detailed_balance else 0,
                "staked": float(detailed_balance.staked) if detailed_balance else 0,
                "total": float(detailed_balance.total) if detailed_balance else 0,
                "context_allocated": float(detailed_balance.context_allocated) if detailed_balance else 0
            } if detailed_balance else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get balance for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/faucet/request")
async def request_testnet_matic(
    request: FaucetRequest,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Request testnet MATIC from faucet
    """
    try:
        async with PolygonFaucetIntegration() as faucet:
            result = await faucet.request_matic(
                request.wallet_address,
                request.preferred_faucet
            )
        
        # Broadcast faucet result to user's WebSocket connections
        await connection_manager.broadcast_to_user(user_id, {
            "type": "faucet_request",
            "result": {
                "status": result.status.value,
                "transaction_hash": result.transaction_hash,
                "amount": result.amount,
                "message": result.message
            }
        })
        
        return {
            "success": result.status.value in ["success", "pending"],
            "status": result.status.value,
            "transaction_hash": result.transaction_hash,
            "amount": result.amount,
            "message": result.message,
            "retry_after": result.retry_after
        }
        
    except Exception as e:
        logger.error(f"Faucet request failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/faucet/status")
async def get_faucet_status(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get status of all available faucets
    """
    try:
        async with PolygonFaucetIntegration() as faucet:
            status = await faucet.get_faucet_status()
            urls = faucet.get_faucet_urls()
        
        return {
            "faucets": status,
            "manual_urls": urls
        }
        
    except Exception as e:
        logger.error(f"Failed to get faucet status for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transactions/{address}")
async def get_transaction_history(
    address: str,
    from_block: int = 0,
    limit: int = 50,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get transaction history for address
    """
    try:
        interface = await connection_manager.get_user_contract_interface(user_id)
        if not interface:
            raise HTTPException(status_code=400, detail="Contract interface not initialized")
        
        # Get transfer events
        events = await interface.get_transfer_events(
            from_block=from_block,
            address_filter=address
        )
        
        # Limit results
        events = events[-limit:] if len(events) > limit else events
        
        return {
            "address": address,
            "transactions": events,
            "total": len(events)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get transaction history for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time Web3 updates
    
    🔐 SECURITY:
    Requires JWT authentication for Web3 operations. Only authenticated users
    can receive real-time wallet balance updates and transaction notifications.
    """
    from prsm.interface.api.websocket_auth import authenticate_websocket_connection, cleanup_websocket_connection, WebSocketAuthError, require_websocket_permission
    
    try:
        # 🛡️ AUTHENTICATE CONNECTION BEFORE ACCEPTING
        connection = await authenticate_websocket_connection(websocket, user_id, "web3")
        await websocket.accept()
        await connection_manager.add_websocket_connection(user_id, websocket)
        
        logger.info("Secure Web3 WebSocket connection established",
                   user_id=user_id,
                   username=connection.username,
                   role=connection.role.value,
                   ip_address=connection.ip_address)
        
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Secure Web3 WebSocket connected",
            "user": connection.username,
            "permissions": connection.permissions,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for incoming messages
                data_text = await websocket.receive_text()
                
                # 🛡️ VALIDATE MESSAGE SIZE AND RATE LIMITS
                from prsm.core.security import validate_websocket_message
                try:
                    await validate_websocket_message(websocket, data_text, user_id)
                except Exception as e:
                    logger.warning("Web3 WebSocket message validation failed",
                                 user_id=user_id,
                                 error=str(e))
                    await websocket.close(code=1008, reason="Message validation failed")
                    return
                
                data = json.loads(data_text)
                
                # Handle different message types
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif data.get("type") == "subscribe_balance":
                    # Start balance monitoring (requires wallet.read permission)
                    try:
                        await require_websocket_permission(websocket, "wallet.read")
                        await websocket.send_json({
                            "type": "balance_subscription",
                            "status": "active",
                            "message": "Balance monitoring enabled"
                        })
                    except WebSocketAuthError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Permission denied: wallet.read required for balance monitoring"
                        })
                elif data.get("type") == "subscribe_transactions":
                    # Start transaction monitoring (requires wallet.read permission)
                    try:
                        await require_websocket_permission(websocket, "wallet.read")
                        await websocket.send_json({
                            "type": "transaction_subscription",
                            "status": "active",
                            "message": "Transaction monitoring enabled"
                        })
                    except WebSocketAuthError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Permission denied: wallet.read required for transaction monitoring"
                        })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Web3 WebSocket error for user {user_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketAuthError as e:
        # Authentication failed - close with appropriate code
        logger.warning("Web3 WebSocket authentication failed",
                      user_id=user_id,
                      error=e.message,
                      code=e.code)
        await websocket.close(code=e.code, reason=e.message)
        return
        
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.remove_websocket_connection(user_id, websocket)
        await cleanup_websocket_connection(websocket)
        
        # 🛡️ CLEANUP SECURITY TRACKING
        from prsm.core.security import cleanup_websocket_connection as cleanup_security
        await cleanup_security(websocket)

@router.get("/gas/estimate")
async def estimate_gas_price(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current gas price estimates
    """
    try:
        connector = await connection_manager.get_user_connector(user_id)
        if not connector or not connector.is_connected:
            raise HTTPException(status_code=400, detail="Wallet not connected")
        
        gas_prices = await connector.get_gas_price()
        
        return {
            "gas_prices": gas_prices,
            "network": connector.current_network.value if connector.current_network else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get gas prices for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disconnect")
async def disconnect_wallet(
    user_id: str = Depends(get_current_user)
) -> Dict[str, bool]:
    """
    Disconnect Web3 wallet
    """
    try:
        # Remove connector and contract interface
        if user_id in connection_manager.connections:
            connection_manager.connections[user_id].disconnect()
            del connection_manager.connections[user_id]
            
        if user_id in connection_manager.contracts:
            del connection_manager.contracts[user_id]
        
        # Broadcast disconnection to user's WebSocket connections
        await connection_manager.broadcast_to_user(user_id, {
            "type": "wallet_disconnected",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Failed to disconnect wallet for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))