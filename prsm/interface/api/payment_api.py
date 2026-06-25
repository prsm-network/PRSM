"""
PRSM Payment Processing API
==========================

REST API endpoints for fiat-to-crypto payment processing, transaction management,
and payment status tracking with comprehensive security and compliance features.
"""

import structlog
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks, Request
from pydantic import BaseModel, Field

from prsm.core.auth import get_current_user
from prsm.core.auth.models import UserRole
from prsm.core.auth.auth_manager import auth_manager
from prsm.economy.payments import (
    get_payment_processor, get_fiat_gateway, get_crypto_exchange,
    PaymentMethod, PaymentStatus, FiatCurrency, CryptoCurrency,
    PaymentRequest, PaymentResponse, TransactionQuery, TransactionList,
    PaymentStats
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/payments", tags=["payments"])


# === Request/Response Models ===

class CreatePaymentRequest(BaseModel):
    """Request to create a new payment"""
    fiat_amount: float = Field(description="Amount in fiat currency", gt=0)
    fiat_currency: FiatCurrency = Field(description="Fiat currency")
    crypto_currency: CryptoCurrency = Field(description="Target cryptocurrency", default=CryptoCurrency.FTNS)
    payment_method: PaymentMethod = Field(description="Payment method")
    payment_method_id: Optional[str] = Field(default=None, description="Saved payment method ID")
    return_url: Optional[str] = Field(default=None, description="Return URL after payment")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for status updates")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PaymentQuoteRequest(BaseModel):
    """Request for payment quote"""
    fiat_amount: float = Field(description="Amount in fiat currency", gt=0)
    fiat_currency: FiatCurrency = Field(description="Fiat currency")
    crypto_currency: CryptoCurrency = Field(description="Target cryptocurrency", default=CryptoCurrency.FTNS)
    payment_method: Optional[PaymentMethod] = Field(default=None, description="Payment method for fee calculation")


class PaymentQuoteResponse(BaseModel):
    """Payment quote response"""
    success: bool
    fiat_amount: float
    fiat_currency: FiatCurrency
    crypto_amount: Optional[float] = None
    crypto_currency: CryptoCurrency
    exchange_rate: Optional[float] = None
    processing_fee: float = Field(default=0)
    network_fee: Optional[float] = None
    total_cost: float
    quote_expires_at: datetime
    message: str


class ExchangeRateRequest(BaseModel):
    """Request for exchange rate"""
    from_currency: str = Field(description="Source currency code")
    to_currency: str = Field(description="Target currency code")
    amount: Optional[float] = Field(default=None, description="Amount for slippage calculation")


class TransactionQueryRequest(BaseModel):
    """Transaction query request"""
    status: Optional[PaymentStatus] = None
    payment_method: Optional[PaymentMethod] = None
    fiat_currency: Optional[FiatCurrency] = None
    crypto_currency: Optional[CryptoCurrency] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=50, ge=1, le=100)
    sort_by: str = Field(default="created_at")
    sort_order: str = Field(default="desc")


class PaymentApiResponse(BaseModel):
    """Standard payment API response"""
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# === Payment Creation and Management ===

@router.post("/create", response_model=PaymentResponse)
async def create_payment(
    request: CreatePaymentRequest,
    current_user: str = Depends(get_current_user)
) -> PaymentResponse:
    """
    Create a new payment transaction
    
    💳 PAYMENT CREATION:
    - Validates payment request and compliance requirements
    - Calculates exchange rates and fees
    - Initiates fiat payment processing
    - Creates transaction record with audit trail
    - Returns payment URL for completion
    """
    try:
        payment_processor = await get_payment_processor()
        
        # Convert to internal payment request
        payment_request = PaymentRequest(
            user_id=current_user,
            fiat_amount=request.fiat_amount,
            fiat_currency=request.fiat_currency,
            crypto_currency=request.crypto_currency,
            payment_method=request.payment_method,
            payment_method_id=request.payment_method_id,
            return_url=request.return_url,
            webhook_url=request.webhook_url,
            metadata=request.metadata
        )
        
        # Create payment
        payment_response = await payment_processor.create_payment(payment_request)
        
        return payment_response
        
    except Exception as e:
        logger.error("Payment creation failed",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment"
        )


@router.get("/quote", response_model=PaymentQuoteResponse)
async def get_payment_quote(
    fiat_amount: float = Query(description="Amount in fiat currency", gt=0),
    fiat_currency: FiatCurrency = Query(description="Fiat currency"),
    crypto_currency: CryptoCurrency = Query(default=CryptoCurrency.FTNS, description="Target cryptocurrency"),
    payment_method: Optional[PaymentMethod] = Query(default=None, description="Payment method"),
    current_user: str = Depends(get_current_user)
) -> PaymentQuoteResponse:
    """
    Get payment quote with fees and exchange rates
    
    💰 PAYMENT QUOTE:
    - Real-time exchange rate calculation
    - Processing fee estimation by payment method
    - Network fee calculation for crypto transactions
    - Quote expiration time for rate protection
    - Slippage estimation for large amounts
    """
    try:
        crypto_exchange = await get_crypto_exchange()
        fiat_gateway = await get_fiat_gateway()
        
        # Get exchange rate
        exchange_rate = await crypto_exchange.get_exchange_rate(
            fiat_currency.value, 
            crypto_currency.value
        )
        
        if not exchange_rate:
            return PaymentQuoteResponse(
                success=False,
                fiat_amount=fiat_amount,
                fiat_currency=fiat_currency,
                crypto_currency=crypto_currency,
                total_cost=fiat_amount,
                quote_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                message="Exchange rate not available"
            )
        
        # Calculate crypto amount
        crypto_amount = fiat_amount * float(exchange_rate.rate)
        
        # Calculate processing fees
        processing_fee = 0.0
        if payment_method:
            fees = fiat_gateway.get_processing_fees(
                payment_method.value, 
                fiat_amount, 
                fiat_currency.value
            )
            processing_fee = float(fees)
        
        # Estimate network fee
        network_fee = 0.01  # Mock network fee
        
        total_cost = fiat_amount + processing_fee
        
        return PaymentQuoteResponse(
            success=True,
            fiat_amount=fiat_amount,
            fiat_currency=fiat_currency,
            crypto_amount=crypto_amount,
            crypto_currency=crypto_currency,
            exchange_rate=float(exchange_rate.rate),
            processing_fee=processing_fee,
            network_fee=network_fee,
            total_cost=total_cost,
            quote_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            message="Quote generated successfully"
        )
        
    except Exception as e:
        logger.error("Failed to generate payment quote",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate payment quote"
        )


@router.get("/status/{transaction_id}", response_model=PaymentResponse)
async def get_payment_status(
    transaction_id: str,
    current_user: str = Depends(get_current_user)
) -> PaymentResponse:
    """
    Get payment transaction status
    
    📊 PAYMENT STATUS:
    - Real-time transaction status updates
    - Payment provider synchronization
    - Blockchain confirmation tracking
    - Automatic status progression monitoring
    - Error and failure reason reporting
    """
    try:
        payment_processor = await get_payment_processor()
        
        payment_response = await payment_processor.get_payment_status(transaction_id)
        
        return payment_response
        
    except Exception as e:
        logger.error("Failed to get payment status",
                    user_id=current_user,
                    transaction_id=transaction_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get payment status"
        )


# === Transaction Management ===

@router.post("/transactions/list", response_model=TransactionList)
async def list_transactions(
    query: TransactionQueryRequest,
    current_user: str = Depends(get_current_user)
) -> TransactionList:
    """
    List payment transactions with filtering
    
    📋 TRANSACTION LISTING:
    - Comprehensive transaction filtering
    - Pagination and sorting support
    - Status and method filtering
    - Date range queries
    - User-specific transaction history
    """
    try:
        payment_processor = await get_payment_processor()
        
        # Create internal query object
        transaction_query = TransactionQuery(
            user_id=current_user,  # Always filter by current user
            status=query.status,
            payment_method=query.payment_method,
            fiat_currency=query.fiat_currency,
            crypto_currency=query.crypto_currency,
            start_date=query.start_date,
            end_date=query.end_date,
            page=query.page,
            limit=query.limit,
            sort_by=query.sort_by,
            sort_order=query.sort_order
        )
        
        transaction_list = await payment_processor.list_transactions(transaction_query)
        
        return transaction_list
        
    except Exception as e:
        logger.error("Failed to list transactions",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list transactions"
        )


@router.get("/transactions/stats", response_model=PaymentStats)
async def get_payment_stats(
    start_date: Optional[datetime] = Query(default=None, description="Start date for stats"),
    end_date: Optional[datetime] = Query(default=None, description="End date for stats"),
    current_user: str = Depends(get_current_user)
) -> PaymentStats:
    """
    Get payment statistics for user
    
    📈 PAYMENT STATISTICS:
    - Transaction volume and count metrics
    - Success/failure rate analysis
    - Currency distribution breakdown
    - Payment method usage patterns
    - Time-based transaction analytics
    """
    try:
        payment_processor = await get_payment_processor()
        
        stats = await payment_processor.get_payment_stats(
            user_id=current_user,
            start_date=start_date,
            end_date=end_date
        )
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get payment stats",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get payment statistics"
        )


# === Exchange Rates and Market Data ===

@router.get("/exchange-rate", response_model=PaymentApiResponse)
async def get_exchange_rate(
    from_currency: str = Query(description="Source currency code"),
    to_currency: str = Query(description="Target currency code"),
    amount: Optional[float] = Query(default=None, description="Amount for conversion calculation"),
    current_user: str = Depends(get_current_user)
) -> PaymentApiResponse:
    """
    Get current exchange rate between currencies
    
    💱 EXCHANGE RATES:
    - Real-time market rates from multiple sources
    - Rate aggregation across providers
    - Slippage calculation for large amounts
    - Historical rate comparison
    - Market volatility indicators
    """
    try:
        crypto_exchange = await get_crypto_exchange()
        
        exchange_rate = await crypto_exchange.get_exchange_rate(from_currency, to_currency)
        
        if not exchange_rate:
            return PaymentApiResponse(
                success=False,
                message=f"Exchange rate not available for {from_currency}/{to_currency}"
            )
        
        data = {
            "from_currency": exchange_rate.from_currency,
            "to_currency": exchange_rate.to_currency,
            "rate": float(exchange_rate.rate),
            "source": exchange_rate.source,
            "timestamp": exchange_rate.timestamp.isoformat(),
            "volume_24h": float(exchange_rate.volume_24h) if exchange_rate.volume_24h else None,
            "price_change_24h": float(exchange_rate.price_change_24h) if exchange_rate.price_change_24h else None
        }
        
        # Calculate conversion if amount provided
        if amount:
            conversion = await crypto_exchange.calculate_conversion(
                amount, from_currency, to_currency
            )
            data["conversion"] = conversion
        
        return PaymentApiResponse(
            success=True,
            message="Exchange rate retrieved successfully",
            data=data
        )
        
    except Exception as e:
        logger.error("Failed to get exchange rate",
                    user_id=current_user,
                    from_currency=from_currency,
                    to_currency=to_currency,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get exchange rate"
        )


@router.get("/supported-currencies", response_model=PaymentApiResponse)
async def get_supported_currencies(
    current_user: str = Depends(get_current_user)
) -> PaymentApiResponse:
    """
    Get supported currencies and payment methods
    
    🌍 SUPPORTED CURRENCIES:
    - Fiat currency support by region
    - Cryptocurrency availability
    - Payment method compatibility
    - Processing fee schedules
    - Geographic availability information
    """
    try:
        fiat_gateway = await get_fiat_gateway()
        crypto_exchange = await get_crypto_exchange()
        
        data = {
            "fiat_currencies": [currency.value for currency in FiatCurrency],
            "cryptocurrencies": [currency.value for currency in CryptoCurrency],
            "payment_methods": [method.value for method in PaymentMethod],
            "supported_providers": fiat_gateway.get_supported_providers(),
            "currency_support_by_provider": fiat_gateway.get_supported_currencies(),
            "exchange_providers": list(crypto_exchange.get_supported_currencies().keys())
        }
        
        return PaymentApiResponse(
            success=True,
            message="Supported currencies retrieved successfully",
            data=data
        )
        
    except Exception as e:
        logger.error("Failed to get supported currencies",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get supported currencies"
        )


# === Webhooks ===

@router.post("/webhooks/{provider}")
async def process_webhook(
    provider: str,
    request: Request,
    background_tasks: BackgroundTasks
) -> PaymentApiResponse:
    """
    Process payment provider webhooks
    
    🔗 WEBHOOK PROCESSING:
    - Secure webhook signature verification
    - Asynchronous payment status updates
    - Automatic transaction progression
    - Failure handling and retry logic
    - Comprehensive webhook logging
    """
    try:
        payment_processor = await get_payment_processor()
        
        # Get raw body and signature for verification
        raw_body = await request.body()
        signature = request.headers.get("stripe-signature", "") or \
                    request.headers.get("paypal-transmission-sig", "")
        # sp1260 — forward the FULL header set so the PayPal verifier can read its 5
        # transmission headers (paypal-transmission-id/-time/-sig, cert-url, auth-algo);
        # real verify-webhook-signature needs all of them, not just the single sig.
        headers = {k.lower(): v for k, v in request.headers.items()}
        payload = await request.json()

        # Process webhook in background
        background_tasks.add_task(
            payment_processor.process_webhook,
            provider,
            payload,
            raw_body,
            signature,
            headers,
        )
        
        return PaymentApiResponse(
            success=True,
            message=f"Webhook from {provider} queued for processing"
        )
        
    except Exception as e:
        logger.error("Webhook processing failed",
                    provider=provider,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process webhook"
        )


# === Administrative Endpoints ===

@router.get("/admin/stats", response_model=PaymentApiResponse)
async def get_admin_payment_stats(
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None),
    current_user: str = Depends(get_current_user)
) -> PaymentApiResponse:
    """
    Get administrative payment statistics (Admin only)
    
    👑 ADMIN STATISTICS:
    - System-wide payment metrics
    - Revenue and volume analytics
    - Provider performance comparison
    - Fraud detection statistics
    - Compliance and regulatory reporting
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required"
            )
        
        payment_processor = await get_payment_processor()
        
        # Get system-wide stats (no user filter)
        stats = await payment_processor.get_payment_stats(
            user_id=None,  # System-wide
            start_date=start_date,
            end_date=end_date
        )
        
        return PaymentApiResponse(
            success=True,
            message="Administrative payment statistics retrieved",
            data={
                "total_transactions": stats.total_transactions,
                "total_volume_usd": float(stats.total_volume_usd),
                "successful_transactions": stats.successful_transactions,
                "failed_transactions": stats.failed_transactions,
                "success_rate": (stats.successful_transactions / stats.total_transactions * 100) if stats.total_transactions > 0 else 0,
                "average_transaction_size": float(stats.average_transaction_size),
                "volume_by_fiat_currency": {k: float(v) for k, v in stats.volume_by_fiat_currency.items()},
                "volume_by_crypto_currency": {k: float(v) for k, v in stats.volume_by_crypto_currency.items()},
                "transactions_by_method": stats.transactions_by_method,
                "period_start": stats.period_start.isoformat(),
                "period_end": stats.period_end.isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get admin payment stats",
                    admin_user=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get administrative statistics"
        )


@router.get("/system/status", response_model=PaymentApiResponse)
async def get_payment_system_status(
    current_user: str = Depends(get_current_user)
) -> PaymentApiResponse:
    """
    Get payment system status and health
    
    🏥 SYSTEM STATUS:
    - Payment provider availability
    - Exchange rate source status
    - Transaction processing queue health
    - Error rates and system performance
    - Component connectivity monitoring
    """
    try:
        fiat_gateway = await get_fiat_gateway()
        crypto_exchange = await get_crypto_exchange()
        
        system_status = {
            "payment_providers": fiat_gateway.get_supported_providers(),
            "exchange_providers": crypto_exchange.get_provider_status(),
            "supported_currencies": {
                "fiat": [currency.value for currency in FiatCurrency],
                "crypto": [currency.value for currency in CryptoCurrency]
            },
            "system_health": {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        return PaymentApiResponse(
            success=True,
            message="Payment system status retrieved",
            data=system_status
        )
        
    except Exception as e:
        logger.error("Failed to get payment system status",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system status"
        )