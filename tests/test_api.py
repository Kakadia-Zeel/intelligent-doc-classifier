"""Integration tests for the FastAPI application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model_service():
    """Create a mock model service for testing."""
    service = MagicMock()
    service.model_type = "lightgbm"
    service.label_names = ["Credit card", "Mortgage", "Debt collection"]
    service.predict.return_value = {
        "predicted_class": "Credit card",
        "confidence": 0.92,
        "probabilities": {
            "Credit card": 0.92,
            "Mortgage": 0.05,
            "Debt collection": 0.03,
        },
    }
    service.predict_batch.return_value = [
        {
            "predicted_class": "Credit card",
            "confidence": 0.92,
            "probabilities": {"Credit card": 0.92, "Mortgage": 0.05, "Debt collection": 0.03},
        },
        {
            "predicted_class": "Mortgage",
            "confidence": 0.88,
            "probabilities": {"Credit card": 0.02, "Mortgage": 0.88, "Debt collection": 0.10},
        },
    ]
    service.get_metrics.return_value = {
        "total_predictions": 100,
        "avg_confidence": 0.85,
        "predictions_per_class": {"Credit card": 40, "Mortgage": 35, "Debt collection": 25},
        "drift_detected": False,
        "drift_details": None,
    }
    return service


@pytest.fixture
def client(mock_model_service):
    """Create a test client with mocked model service."""
    with patch(
        "src.serving.app.get_model_service", return_value=mock_model_service
    ):
        from src.serving.app import app

        with TestClient(app) as client:
            yield client


class TestClassifyEndpoint:
    """Tests for POST /classify."""

    def test_classify_success(self, client):
        response = client.post(
            "/classify",
            json={"text": "I have a problem with my credit card billing."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "Credit card"
        assert data["confidence"] == 0.92
        assert "probabilities" in data

    def test_classify_short_text(self, client):
        response = client.post("/classify", json={"text": "short"})
        assert response.status_code == 422  # Validation error

    def test_classify_missing_text(self, client):
        response = client.post("/classify", json={})
        assert response.status_code == 422

    def test_response_headers(self, client):
        response = client.post(
            "/classify",
            json={"text": "A complaint about my credit card charges."},
        )
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers


class TestBatchEndpoint:
    """Tests for POST /classify/batch."""

    def test_batch_success(self, client):
        response = client.post(
            "/classify/batch",
            json={
                "texts": [
                    "Credit card problem with unauthorized charge.",
                    "My mortgage payment was applied incorrectly.",
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty(self, client):
        response = client.post("/classify/batch", json={"texts": []})
        assert response.status_code == 422


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "drift_detected" in data
