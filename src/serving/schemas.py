"""Pydantic request/response schemas for the classification API."""

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Single document classification request."""

    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="The document text to classify",
        examples=["I have a problem with my credit card billing statement."],
    )
    explain: bool = Field(
        default=False,
        description="Whether to include LIME explanation (slower)",
    )


class ClassifyResponse(BaseModel):
    """Single document classification response."""

    predicted_class: str = Field(description="Predicted product category")
    confidence: float = Field(description="Prediction confidence score")
    probabilities: dict[str, float] = Field(
        description="Probability for each class"
    )
    explanation: dict | None = Field(
        default=None,
        description="LIME explanation with top contributing words",
    )


class BatchClassifyRequest(BaseModel):
    """Batch document classification request."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of document texts to classify",
    )


class BatchClassifyResponse(BaseModel):
    """Batch document classification response."""

    predictions: list[ClassifyResponse] = Field(
        description="List of classification results"
    )
    count: int = Field(description="Number of documents classified")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    model_loaded: bool = Field(description="Whether the ML model is loaded")
    model_type: str = Field(description="Type of model being served")


class MetricsResponse(BaseModel):
    """Model metrics and monitoring response."""

    total_predictions: int
    avg_confidence: float
    predictions_per_class: dict[str, int]
    drift_detected: bool
    drift_details: dict | None = None
