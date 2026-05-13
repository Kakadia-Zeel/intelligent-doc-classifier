"""Tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from src.serving.schemas import (
    BatchClassifyRequest,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
)


class TestClassifyRequest:
    """Tests for ClassifyRequest schema validation."""

    def test_valid_request(self):
        req = ClassifyRequest(text="This is a valid complaint about my credit card.")
        assert req.text == "This is a valid complaint about my credit card."
        assert req.explain is False

    def test_with_explanation(self):
        req = ClassifyRequest(text="A valid complaint text here.", explain=True)
        assert req.explain is True

    def test_too_short_text(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(text="short")

    def test_empty_text(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(text="")


class TestBatchClassifyRequest:
    """Tests for BatchClassifyRequest schema validation."""

    def test_valid_batch(self):
        req = BatchClassifyRequest(texts=["First complaint text", "Second complaint text"])
        assert len(req.texts) == 2

    def test_empty_list(self):
        with pytest.raises(ValidationError):
            BatchClassifyRequest(texts=[])


class TestClassifyResponse:
    """Tests for ClassifyResponse schema."""

    def test_valid_response(self):
        resp = ClassifyResponse(
            predicted_class="Credit card",
            confidence=0.95,
            probabilities={"Credit card": 0.95, "Mortgage": 0.05},
        )
        assert resp.predicted_class == "Credit card"
        assert resp.explanation is None

    def test_response_with_explanation(self):
        resp = ClassifyResponse(
            predicted_class="Mortgage",
            confidence=0.88,
            probabilities={"Mortgage": 0.88},
            explanation={"top_features": [{"word": "mortgage", "weight": 0.5}]},
        )
        assert resp.explanation is not None


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy(self):
        resp = HealthResponse(status="healthy", model_loaded=True, model_type="transformer")
        assert resp.status == "healthy"
