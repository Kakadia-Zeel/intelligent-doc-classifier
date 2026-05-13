"""FastAPI application for document classification."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models.explain import TextExplainer
from src.serving.dependencies import get_model_service
from src.serving.middleware import RequestLoggingMiddleware
from src.serving.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    MetricsResponse,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Loading model on startup")
    try:
        service = get_model_service()
        logger.info("Model loaded successfully", model_type=service.model_type)
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Intelligent Document Classifier",
    description="Production-grade API for classifying consumer financial documents",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify", response_model=ClassifyResponse)
async def classify_document(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a single document into a product category."""
    service = get_model_service()

    try:
        result = service.predict(request.text)
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

    explanation = None
    if request.explain:
        try:
            explainer = TextExplainer(
                predict_fn=service.get_predict_fn(),
                class_names=service.label_names,
            )
            explanation = explainer.explain(request.text)
        except Exception as e:
            logger.warning("Explanation generation failed", error=str(e))

    return ClassifyResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        explanation=explanation,
    )


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(request: BatchClassifyRequest) -> BatchClassifyResponse:
    """Classify multiple documents in a single request."""
    service = get_model_service()

    try:
        results = service.predict_batch(request.texts)
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Batch prediction failed") from e

    predictions = [
        ClassifyResponse(
            predicted_class=r["predicted_class"],
            confidence=r["confidence"],
            probabilities=r["probabilities"],
        )
        for r in results
    ]

    return BatchClassifyResponse(predictions=predictions, count=len(predictions))


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check if the service is healthy and model is loaded."""
    try:
        service = get_model_service()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type=service.model_type,
        )
    except Exception:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_type="none",
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get model performance metrics and drift information."""
    service = get_model_service()
    metrics = service.get_metrics()

    return MetricsResponse(
        total_predictions=metrics["total_predictions"],
        avg_confidence=metrics["avg_confidence"],
        predictions_per_class=metrics["predictions_per_class"],
        drift_detected=metrics["drift_detected"],
        drift_details=metrics["drift_details"],
    )
