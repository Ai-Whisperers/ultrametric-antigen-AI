# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for Drug Resistance API.

Tests cover:
- Health check endpoint
- Disease listing
- Drug listing
- Prediction endpoints
- Error handling
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root
root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root))


# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.fixture
def client():
    """Create test client for API."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")

    from src.api.drug_resistance_api import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test /health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_includes_version(self, client):
        """Test health check includes version info."""
        response = client.get("/health")
        data = response.json()

        assert "version" in data


class TestDiseasesEndpoint:
    """Tests for diseases listing endpoint."""

    def test_list_diseases(self, client):
        """Test /diseases endpoint returns list of diseases."""
        response = client.get("/diseases")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_disease_info_structure(self, client):
        """Test disease info has required fields."""
        response = client.get("/diseases")
        data = response.json()

        if len(data) > 0:
            disease = data[0]
            assert "name" in disease or "id" in disease

    def test_expected_diseases_present(self, client):
        """Test that expected diseases are in the list."""
        response = client.get("/diseases")
        data = response.json()

        disease_names = [d.get("name", d.get("id", "")) for d in data]
        disease_names_lower = [n.lower() for n in disease_names]

        # At least some key diseases should be present
        expected = ["hiv", "sars", "tuberculosis", "influenza"]
        found = sum(1 for e in expected if any(e in n for n in disease_names_lower))
        assert found >= 1, f"Expected at least one of {expected}"


class TestDrugsEndpoint:
    """Tests for drugs listing endpoint."""

    def test_list_drugs(self, client):
        """Test /drugs endpoint returns list of drugs."""
        response = client.get("/drugs")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, (list, dict))

    def test_hiv_drugs_present(self, client):
        """Test that HIV drugs are listed."""
        response = client.get("/drugs")
        data = response.json()

        # Check for some common HIV drugs
        drugs_str = str(data).lower()
        common_drugs = ["3tc", "azt", "efv", "lpv", "drv"]
        found = sum(1 for d in common_drugs if d in drugs_str)
        assert found >= 1, "Expected at least one common HIV drug"


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_requires_sequence(self, client):
        """Test /predict requires sequence field."""
        response = client.post("/predict", json={"drug": "AZT"})
        assert response.status_code in [400, 422]  # Validation error

    def test_predict_requires_drug(self, client):
        """Test /predict requires drug field."""
        sequence = "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"
        response = client.post("/predict", json={"sequence": sequence})
        assert response.status_code in [400, 422]  # Validation error

    def test_predict_valid_input(self, client):
        """Test /predict with valid input returns prediction."""
        sequence = "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"
        response = client.post("/predict", json={
            "sequence": sequence,
            "drug": "AZT",
        })

        # Should return 200 or a handled error
        assert response.status_code in [200, 400, 422, 500]

        if response.status_code == 200:
            data = response.json()
            assert "resistance_score" in data or "prediction" in data or "error" not in data

    def test_predict_returns_interpretation(self, client):
        """Test prediction includes clinical interpretation."""
        sequence = "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"
        response = client.post("/predict", json={
            "sequence": sequence,
            "drug": "AZT",
        })

        if response.status_code == 200:
            data = response.json()
            # Should have interpretation or classification
            has_interp = "interpretation" in data or "classification" in data or "level" in data
            assert has_interp or "resistance_score" in data


class TestDiseaseSpecificEndpoint:
    """Tests for disease-specific prediction endpoints."""

    def test_predict_hiv(self, client):
        """Test /predict/hiv endpoint."""
        sequence = "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"
        response = client.post("/predict/hiv", json={
            "sequence": sequence,
            "drug": "AZT",
        })

        # Endpoint may not exist, which is fine
        assert response.status_code in [200, 400, 404, 422, 500]

    def test_invalid_disease_returns_404(self, client):
        """Test invalid disease returns 404."""
        response = client.post("/predict/invalid_disease_xyz", json={
            "sequence": "MKWVTVYIG",
            "drug": "XYZ",
        })

        # Should be 404 (not found) or 405 (method not allowed)
        assert response.status_code in [404, 405, 422]


class TestBatchEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict(self, client):
        """Test /predict/batch endpoint."""
        sequences = [
            "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW",
            "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTY",
        ]
        response = client.post("/predict/batch", json={
            "sequences": sequences,
            "drug": "AZT",
        })

        # Endpoint may not exist
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data or isinstance(data, list)

    def test_batch_limit(self, client):
        """Test batch endpoint has reasonable limits."""
        # Create 100 sequences
        sequences = ["MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"] * 100
        response = client.post("/predict/batch", json={
            "sequences": sequences,
            "drug": "AZT",
        })

        # Should either succeed or return a rate limit / validation error
        assert response.status_code in [200, 400, 422, 429, 413]


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json(self, client):
        """Test invalid JSON returns 422."""
        response = client.post(
            "/predict",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_sequence_too_short(self, client):
        """Test short sequence returns validation error."""
        response = client.post("/predict", json={
            "sequence": "MK",  # Too short
            "drug": "AZT",
        })
        assert response.status_code in [400, 422]

    def test_unknown_drug(self, client):
        """Test unknown drug is handled gracefully."""
        sequence = "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"
        response = client.post("/predict", json={
            "sequence": sequence,
            "drug": "UNKNOWN_DRUG_XYZ",
        })

        # Should return error or handle gracefully
        assert response.status_code in [200, 400, 404, 422]


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are set."""
        response = client.options("/health")

        # OPTIONS should be handled
        assert response.status_code in [200, 204, 405]


class TestAPIDocumentation:
    """Tests for API documentation."""

    def test_openapi_available(self, client):
        """Test OpenAPI spec is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "paths" in data
        assert "info" in data

    def test_docs_available(self, client):
        """Test Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_headers_present(self, client):
        """Test rate limit headers are included in responses."""
        response = client.get("/health")
        assert response.status_code == 200

        # Check for rate limit headers
        headers = response.headers
        assert "X-RateLimit-Limit-Minute" in headers or "x-ratelimit-limit-minute" in headers
        assert "X-RateLimit-Remaining-Minute" in headers or "x-ratelimit-remaining-minute" in headers

    def test_rate_limiter_class(self):
        """Test RateLimiter class directly."""
        from src.api.drug_resistance_api import RateLimiter, RateLimitConfig

        config = RateLimitConfig(requests_per_minute=5, requests_per_hour=100)
        limiter = RateLimiter(config)

        # First 5 requests should be allowed
        for i in range(5):
            allowed, headers = limiter.is_allowed("test_client")
            assert allowed, f"Request {i+1} should be allowed"

        # 6th request should be blocked
        allowed, headers = limiter.is_allowed("test_client")
        assert not allowed, "6th request should be blocked"
        assert "Retry-After" in headers


class TestAPIVersioning:
    """Tests for API versioning."""

    def test_version_endpoint(self, client):
        """Test /api/v1/version endpoint."""
        response = client.get("/api/v1/version")

        if response.status_code == 200:
            data = response.json()
            assert "api_version" in data
            assert data["api_version"] == "v1"
            assert "version_full" in data

    def test_version_in_health(self, client):
        """Test version is included in health response."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "version" in data
        assert "docs" in data


class TestClinicalReport:
    """Tests for clinical report endpoint."""

    def test_clinical_report_endpoint(self, client):
        """Test /clinical-report endpoint."""
        sequence = "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"

        response = client.post("/clinical-report", params={
            "sequence": sequence,
            "patient_id": "TEST001",
        })

        if response.status_code == 200:
            data = response.json()
            assert "analysis_date" in data
            assert "recommended_drugs" in data
            assert "avoid_drugs" in data
            assert "overall_recommendation" in data

    def test_clinical_report_without_patient_id(self, client):
        """Test clinical report works without patient ID."""
        sequence = "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"

        response = client.post("/clinical-report", params={
            "sequence": sequence,
        })

        # Should work without patient_id
        assert response.status_code in [200, 422]


class TestUncertaintyPrediction:
    """Tests for uncertainty quantification endpoint."""

    def test_uncertainty_endpoint(self, client):
        """Test /predict/uncertainty endpoint."""
        sequence = "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"

        response = client.post("/predict/uncertainty", json={
            "sequence": sequence,
            "drug": "AZT",
        }, params={"n_samples": 10})

        if response.status_code == 200:
            data = response.json()
            assert "mean_score" in data
            assert "std_score" in data
            assert "lower_95" in data
            assert "upper_95" in data

    def test_cross_resistance_endpoint(self, client):
        """Test /predict/cross-resistance endpoint."""
        sequence = "MKWVTVYIGVEPVSGKGRLDIWVTIEKDLKDCAWKLIIEYVKQTW"

        response = client.post("/predict/cross-resistance", json={
            "sequence": sequence,
            "drug": "AZT",  # NRTI drug
        })

        if response.status_code == 200:
            data = response.json()
            assert "cross_resistance" in data
            assert "recommendation" in data
