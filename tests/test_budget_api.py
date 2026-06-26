"""
API Endpoint Tests for FTNS Budget Management

🌐 API TEST COVERAGE:
- Budget creation and prediction endpoints
- Real-time status monitoring endpoints
- Spending and reservation endpoints
- Budget expansion and approval workflows
- Analytics and reporting endpoints
- Error handling and validation
"""

import pytest
import asyncio
import json
from decimal import Decimal
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

# Import the FastAPI app and budget components
from prsm.interface.api.main import app
from prsm.interface.api.budget_api import get_budget_manager
from prsm.economy.tokenomics.ftns_budget_manager import FTNSBudgetManager, SpendingCategory


class TestBudgetAPI:
    """Test suite for Budget Management API endpoints"""

    @pytest.fixture(autouse=True)
    def _override_auth(self):
        """Bypass FastAPI auth dependency for all tests in this class."""
        from prsm.interface.auth import get_current_user as _get_current_user
        app.dependency_overrides[_get_current_user] = lambda: {
            "user_id": "test_api_user_001",
            "username": "test_user",
        }
        yield
        app.dependency_overrides.clear()

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user"""
        return {"user_id": "test_api_user_001", "username": "test_user"}

    @pytest.fixture
    def budget_manager_mock(self):
        """Mock budget manager for isolated API testing"""
        return AsyncMock(spec=FTNSBudgetManager)
    
    def test_predict_cost_endpoint(self, client):
        """Test cost prediction endpoint"""
        print("\n🔮 Testing Cost Prediction API...")
        
        # Mock the authentication and budget manager
        with patch('prsm.interface.api.budget_api.get_current_user') as mock_auth, \
             patch('prsm.interface.api.budget_api.get_ftns_budget_manager') as mock_manager:
            
            # Setup mocks
            mock_auth.return_value = {"user_id": "test_user"}
            mock_budget_manager = AsyncMock()
            mock_manager.return_value = mock_budget_manager
            
            # Mock prediction response — use MagicMock so we control all attributes
            from unittest.mock import MagicMock
            mock_prediction = MagicMock()
            mock_prediction.prediction_id = "pred-test-001"
            mock_prediction.query_complexity = 0.7
            mock_prediction.estimated_total_cost = Decimal('85.5')
            mock_prediction.category_estimates = {
                SpendingCategory.MODEL_INFERENCE: Decimal('51.3'),
                SpendingCategory.AGENT_EXECUTION: Decimal('25.6'),
                SpendingCategory.AGENT_COORDINATION: Decimal('8.6')
            }
            mock_prediction.confidence_score = 0.82
            mock_prediction.get_recommended_budget.return_value = Decimal('95.0')
            mock_prediction.prediction_factors = {"complexity_score": 0.7}
            mock_budget_manager.predict_session_cost.return_value = mock_prediction
            
            # Make API request
            response = client.post(
                "/api/v1/budget/predict-cost",
                params={"prompt": "Analyze quantum field interactions for APM development"}
            )
            
            # Validate response
            assert response.status_code == 200
            data = response.json()
            
            assert "estimated_total_cost" in data
            assert "confidence_score" in data
            assert "recommended_budget" in data
            assert "category_estimates" in data
            
            print(f"✅ Prediction API working: {data['estimated_total_cost']:.1f} FTNS estimated")
            print(f"   Confidence: {data['confidence_score']:.2f}")
            
            return data
    
    def test_create_budget_endpoint(self, client):
        """Test budget creation endpoint"""
        print("\n💳 Testing Budget Creation API...")
        
        with patch('prsm.interface.api.budget_api.get_current_user') as mock_auth, \
             patch('prsm.interface.api.budget_api.get_ftns_budget_manager') as mock_manager:
            
            # Setup mocks
            mock_auth.return_value = {"user_id": "test_user"}
            mock_budget_manager = AsyncMock()
            mock_manager.return_value = mock_budget_manager
            
            # Mock budget creation
            from prsm.economy.tokenomics.ftns_budget_manager import FTNSBudget, BudgetStatus
            from uuid import uuid4
            
            mock_budget = AsyncMock()
            mock_budget.budget_id = uuid4()
            mock_budget.session_id = uuid4()
            mock_budget.total_budget = Decimal('150.0')
            mock_budget.status = BudgetStatus.ACTIVE
            
            mock_budget_manager.create_session_budget.return_value = mock_budget
            mock_budget_manager.get_budget_status.return_value = {
                "budget_id": str(mock_budget.budget_id),
                "session_id": str(mock_budget.session_id),
                "status": "active",
                "total_budget": 150.0,
                "total_spent": 0.0,
                "available_budget": 150.0,
                "utilization_percentage": 0.0,
                "category_breakdown": {},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
            
            # Make API request
            request_data = {
                "prompt": "Complex quantum analysis for APM development",
                "total_budget": 150.0,
                "auto_expand_enabled": True,
                "max_auto_expand": 50.0,
                "expansion_increment": 25.0
            }
            
            response = client.post("/api/v1/budget/create", json=request_data)
            
            # Validate response
            assert response.status_code == 200
            data = response.json()
            
            assert "budget_id" in data
            assert "total_budget" in data
            assert data["total_budget"] == 150.0
            assert data["status"] == "active"
            
            print(f"✅ Budget creation API working: {data['budget_id']}")
            print(f"   Total budget: {data['total_budget']:.1f} FTNS")
            
            return data
    
    def test_budget_status_endpoint(self, client):
        """Test budget status monitoring endpoint"""
        print("\n📊 Testing Budget Status API...")
        
        with patch('prsm.interface.api.budget_api.get_current_user') as mock_auth, \
             patch('prsm.interface.api.budget_api.get_ftns_budget_manager') as mock_manager:
            
            # Setup mocks
            mock_auth.return_value = {"user_id": "test_user"}
            mock_budget_manager = AsyncMock()
            mock_manager.return_value = mock_budget_manager
            
            # Mock budget status
            from uuid import uuid4
            budget_id = uuid4()
            
            mock_budget_manager.get_budget_status.return_value = {
                "budget_id": str(budget_id),
                "user_id": "test_api_user_001",   # sp1282 — caller (dependency-override user) owns this budget
                "session_id": str(uuid4()),
                "status": "active",
                "total_budget": 100.0,
                "total_spent": 35.0,
                "available_budget": 65.0,
                "utilization_percentage": 35.0,
                "category_breakdown": {
                    "model_inference": {
                        "allocated": 60.0,
                        "spent": 25.0,
                        "available": 35.0,
                        "utilization": 41.7
                    },
                    "tool_execution": {
                        "allocated": 30.0,
                        "spent": 10.0,
                        "available": 20.0,
                        "utilization": 33.3
                    }
                },
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:05:30Z"
            }
            
            # Make API request
            response = client.get(f"/api/v1/budget/status/{budget_id}")
            
            # Validate response
            assert response.status_code == 200
            data = response.json()
            
            assert data["budget_id"] == str(budget_id)
            assert data["utilization_percentage"] == 35.0
            assert "category_breakdown" in data
            
            print(f"✅ Status API working: {data['utilization_percentage']:.1f}% utilized")
            print(f"   Available: {data['available_budget']:.1f} FTNS")
            
            return data
    
    def test_spending_endpoint(self, client):
        """Test budget spending tracking endpoint"""
        print("\n💰 Testing Spending API...")
        
        with patch('prsm.interface.api.budget_api.get_current_user') as mock_auth, \
             patch('prsm.interface.api.budget_api.get_ftns_budget_manager') as mock_manager:
            
            # Setup mocks
            mock_auth.return_value = {"user_id": "test_user"}
            mock_budget_manager = AsyncMock()
            mock_manager.return_value = mock_budget_manager
            
            # Mock spending success
            mock_budget_manager.spend_budget_amount.return_value = True
            mock_budget_manager.get_budget_status.return_value = {
                "user_id": "test_api_user_001",   # sp1282 — caller (dependency-override user) owns this budget
                "available_budget": 75.0,
                "utilization_percentage": 25.0
            }
            
            # Make API request
            from uuid import uuid4
            budget_id = uuid4()
            
            request_data = {
                "amount": 25.0,
                "category": "model_inference",
                "description": "LLM inference for quantum analysis"
            }
            
            response = client.post(f"/api/v1/budget/spend/{budget_id}", json=request_data)
            
            # Validate response
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] == True
            assert data["amount_spent"] == 25.0
            assert data["category"] == "model_inference"
            assert "remaining_budget" in data
            
            print(f"✅ Spending API working: {data['amount_spent']:.1f} FTNS spent")
            print(f"   Remaining: {data['remaining_budget']:.1f} FTNS")
            
            return data
    
    def test_budget_expansion_endpoint(self, client):
        """Test budget expansion request endpoint"""
        print("\n📈 Testing Budget Expansion API...")
        
        with patch('prsm.interface.api.budget_api.get_current_user') as mock_auth, \
             patch('prsm.interface.api.budget_api.get_ftns_budget_manager') as mock_manager:
            
            # Setup mocks
            mock_auth.return_value = {"user_id": "test_user"}
            mock_budget_manager = AsyncMock()
            mock_manager.return_value = mock_budget_manager
            
            # Mock expansion request
            from prsm.economy.tokenomics.ftns_budget_manager import BudgetExpandRequest
            from uuid import uuid4
            
            budget_id = uuid4()
            request_id = uuid4()
            
            mock_expand_request = AsyncMock()
            mock_expand_request.request_id = request_id
            mock_expand_request.budget_id = budget_id
            mock_expand_request.requested_amount = Decimal('50.0')
            mock_expand_request.current_utilization = 85.0
            mock_expand_request.remaining_budget = Decimal('15.0')
            mock_expand_request.auto_generated = False
            mock_expand_request.approved = None  # Pending
            from datetime import datetime, timezone as _tz
            mock_expand_request.expires_at = datetime(2024, 1, 1, 1, 0, 0, tzinfo=_tz.utc)
            
            mock_budget_manager.request_budget_expansion.return_value = mock_expand_request
            # sp1282 — ownership gate reads get_budget_status; caller owns this budget
            mock_budget_manager.get_budget_status.return_value = {
                "user_id": "test_api_user_001",   # sp1282 — dependency-override user owns it
                "available_budget": 15.0,
                "utilization_percentage": 85.0,
            }

            # Make API request
            request_data = {
                "requested_amount": 50.0,
                "reason": "Need additional budget for deep analysis",
                "cost_breakdown": {
                    "model_inference": 35.0,
                    "tool_execution": 15.0
                }
            }
            
            response = client.post(f"/api/v1/budget/expand/{budget_id}", json=request_data)
            
            # Validate response
            assert response.status_code == 200
            data = response.json()
            
            assert "request_id" in data
            assert data["requested_amount"] == 50.0
            assert data["current_utilization"] == 85.0
            assert "expires_at" in data
            
            print(f"✅ Expansion API working: {data['requested_amount']:.1f} FTNS requested")
            print(f"   Request ID: {data['request_id']}")
            
            return data
    
    def test_expansion_approval_endpoint(self, client):
        """Test budget expansion approval endpoint"""
        print("\n👤 Testing Expansion Approval API...")
        
        with patch('prsm.interface.api.budget_api.get_current_user') as mock_auth, \
             patch('prsm.interface.api.budget_api.get_ftns_budget_manager') as mock_manager:
            
            # Setup mocks
            mock_auth.return_value = {"user_id": "test_user"}
            mock_budget_manager = AsyncMock()
            mock_manager.return_value = mock_budget_manager
            
            # Mock approval success
            mock_budget_manager.approve_budget_expansion.return_value = True
            
            # Make API request
            from uuid import uuid4
            request_id = uuid4()
            
            request_data = {
                "approved": True,
                "approved_amount": 45.0,
                "reason": "Approved for critical analysis completion"
            }
            
            response = client.post(f"/api/v1/budget/approve-expansion/{request_id}", json=request_data)
            
            # Validate response
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] == True
            assert data["approved"] == True
            assert data["approved_amount"] == 45.0
            
            print(f"✅ Approval API working: {data['approved_amount']:.1f} FTNS approved")
            
            return data
    
    def test_user_budgets_endpoint(self, client):
        """Test user budgets listing endpoint"""
        print("\n📋 Testing User Budgets API...")
        
        with patch('prsm.interface.api.budget_api.get_current_user') as mock_auth, \
             patch('prsm.interface.api.budget_api.get_ftns_budget_manager') as mock_manager:
            
            # Setup mocks
            mock_auth.return_value = {"user_id": "test_user"}
            mock_budget_manager = AsyncMock()
            mock_manager.return_value = mock_budget_manager
            
            # Mock budget manager with sample budgets
            from uuid import uuid4
            
            mock_budget_1 = AsyncMock()
            mock_budget_1.user_id = "test_user"
            mock_budget_1.status.value = "active"
            mock_budget_1.budget_id = uuid4()
            
            mock_budget_2 = AsyncMock()
            mock_budget_2.user_id = "test_user"
            mock_budget_2.status.value = "completed"
            mock_budget_2.budget_id = uuid4()
            
            mock_budget_manager.active_budgets = {mock_budget_1.budget_id: mock_budget_1}
            mock_budget_manager.budget_history = {mock_budget_2.budget_id: mock_budget_2}
            
            # Mock budget status returns
            mock_budget_manager.get_budget_status.side_effect = [
                {
                    "budget_id": str(mock_budget_1.budget_id),
                    "status": "active",
                    "total_budget": 100.0,
                    "created_at": "2024-01-01T00:00:00Z"
                },
                {
                    "budget_id": str(mock_budget_2.budget_id),
                    "status": "completed",
                    "total_budget": 75.0,
                    "created_at": "2024-01-01T00:00:00Z"
                }
            ]
            
            # Make API request
            response = client.get("/api/v1/budget/user-budgets?limit=10")
            
            # Validate response
            assert response.status_code == 200
            data = response.json()
            
            assert "budgets" in data
            assert "total_count" in data
            assert len(data["budgets"]) >= 0
            
            print(f"✅ User budgets API working: {data['total_count']} budgets found")
            
            return data
    
    def test_analytics_endpoint(self, client):
        """Test budget analytics endpoint"""
        print("\n📊 Testing Budget Analytics API...")
        
        with patch('prsm.interface.api.budget_api.get_current_user') as mock_auth, \
             patch('prsm.interface.api.budget_api.get_ftns_budget_manager') as mock_manager:
            
            # Setup mocks
            mock_auth.return_value = {"user_id": "test_user"}
            mock_budget_manager = AsyncMock()
            mock_manager.return_value = mock_budget_manager
            
            from uuid import uuid4
            budget_id = uuid4()
            
            # Mock budget status (include available_budget which the handler requires)
            mock_budget_manager.get_budget_status.return_value = {
                "budget_id": str(budget_id),
                "status": "active",
                "total_budget": 100.0,
                "total_spent": 45.0,
                "available_budget": 55.0,
                "utilization_percentage": 45.0,
                "triggered_alerts": [],
                "pending_expansions": 0,
                "category_breakdown": {
                    "model_inference": {"spent": 30.0},
                    "agent_execution": {"spent": 15.0}
                }
            }
            
            # Mock budget object for detailed analytics
            mock_budget = AsyncMock()
            mock_budget.spending_history = [
                {"action": "spend", "amount": 30.0, "category": "model_inference"},
                {"action": "spend", "amount": 15.0, "category": "tool_execution"}
            ]
            
            mock_budget_manager.active_budgets = {budget_id: mock_budget}
            
            # Make API request
            response = client.get(f"/api/v1/budget/analytics/{budget_id}")
            
            # Validate response
            assert response.status_code == 200
            data = response.json()
            
            assert "budget_overview" in data
            assert "spending_analytics" in data
            assert "category_analysis" in data
            assert "budget_health" in data
            
            print(f"✅ Analytics API working with comprehensive data")
            
            return data
    
    def test_spending_categories_endpoint(self, client):
        """Test spending categories listing endpoint"""
        print("\n📂 Testing Spending Categories API...")
        
        # Make API request (no authentication needed for this endpoint)
        response = client.get("/api/v1/budget/spending-categories")
        
        # Validate response
        assert response.status_code == 200
        data = response.json()
        
        assert "categories" in data
        assert "total_count" in data
        assert data["total_count"] > 0
        
        # Check that expected categories are present (use current SpendingCategory enum values)
        categories = data["categories"]
        assert "model_inference" in categories
        assert "agent_execution" in categories
        assert "agent_coordination" in categories
        
        print(f"✅ Categories API working: {data['total_count']} categories available")
        
        return data
    
    def test_api_error_handling(self, client):
        """Test API error handling scenarios"""
        print("\n❌ Testing API Error Handling...")
        
        # Test invalid budget ID
        from uuid import uuid4
        invalid_budget_id = uuid4()
        
        with patch('prsm.interface.api.budget_api.get_current_user') as mock_auth, \
             patch('prsm.interface.api.budget_api.get_ftns_budget_manager') as mock_manager:
            
            mock_auth.return_value = {"user_id": "test_user"}
            mock_budget_manager = AsyncMock()
            mock_manager.return_value = mock_budget_manager
            
            # Mock budget not found
            mock_budget_manager.get_budget_status.return_value = None
            
            response = client.get(f"/api/v1/budget/status/{invalid_budget_id}")
            assert response.status_code == 404
            
            print("✅ 404 error handling working for invalid budget ID")
        
        # Test invalid request data
        response = client.post("/api/v1/budget/create", json={"invalid": "data"})
        assert response.status_code in [422, 400]  # Validation error
        
        print("✅ Validation error handling working for invalid data")
        
        return True


# API test runner
async def run_api_tests():
    """Run all budget API tests"""
    print("🌐 STARTING BUDGET API TESTS")
    print("=" * 60)
    
    test_api = TestBudgetAPI()
    client = TestClient(app)
    
    try:
        print("\n📡 ENDPOINT FUNCTIONALITY TESTS")
        print("-" * 40)
        
        test_api.test_predict_cost_endpoint(client)
        test_api.test_create_budget_endpoint(client)
        test_api.test_budget_status_endpoint(client)
        test_api.test_spending_endpoint(client)
        test_api.test_budget_expansion_endpoint(client)
        test_api.test_expansion_approval_endpoint(client)
        test_api.test_user_budgets_endpoint(client)
        test_api.test_analytics_endpoint(client)
        test_api.test_spending_categories_endpoint(client)
        
        print("\n🛡️ ERROR HANDLING TESTS")
        print("-" * 40)
        
        test_api.test_api_error_handling(client)
        
        print("\n🎉 ALL API TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n✅ Budget API is operational and ready!")
        print("\nAPI Endpoints Validated:")
        print("• ✅ POST /api/v1/budget/predict-cost - Cost prediction")
        print("• ✅ POST /api/v1/budget/create - Budget creation")
        print("• ✅ GET /api/v1/budget/status/{id} - Real-time monitoring")
        print("• ✅ POST /api/v1/budget/spend/{id} - Spending tracking")
        print("• ✅ POST /api/v1/budget/expand/{id} - Budget expansion")
        print("• ✅ POST /api/v1/budget/approve-expansion/{id} - Expansion approval")
        print("• ✅ GET /api/v1/budget/user-budgets - User budget listing")
        print("• ✅ GET /api/v1/budget/analytics/{id} - Detailed analytics")
        print("• ✅ GET /api/v1/budget/spending-categories - Category listing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run API tests
    success = asyncio.run(run_api_tests())
    exit(0 if success else 1)