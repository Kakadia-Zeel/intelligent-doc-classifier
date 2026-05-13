"""FastAPI dependencies — model loading and inference utilities."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from src.data.preprocess import clean_text
from src.utils.config import load_config

logger = structlog.get_logger(__name__)

ARTIFACTS_DIR = Path("artifacts")


class ModelService:
    """Manages model loading and inference for the serving layer."""

    def __init__(self):
        self.model = None
        self.model_type: str = "none"
        self.label_names: list[str] = []
        self.tfidf_pipeline = None
        self.transformer = None

        # Monitoring state
        self.total_predictions = 0
        self.predictions_per_class: dict[str, int] = defaultdict(int)
        self.confidence_scores: list[float] = []
        self.reference_distribution: Optional[dict] = None

        self._load_reference_distribution()

    def load_model(self) -> None:
        """Load the best available model."""
        serving_config = load_config("serving_config.yaml")
        preferred_type = serving_config["model"]["type"]

        if preferred_type == "transformer":
            try:
                self._load_transformer()
                return
            except Exception as e:
                logger.warning(
                    "Transformer loading failed, falling back to baseline",
                    error=str(e),
                )
                if not serving_config["model"]["fallback_to_baseline"]:
                    raise

        self._load_baseline()

    def _load_transformer(self) -> None:
        """Load the fine-tuned transformer model."""
        from src.models.transformer import TransformerClassifier

        transformer_path = ARTIFACTS_DIR / "transformer"
        if not transformer_path.exists():
            raise FileNotFoundError(f"Transformer not found at {transformer_path}")

        self.transformer = TransformerClassifier.load(transformer_path)
        self.label_names = self.transformer.label_names
        self.model_type = "transformer"
        logger.info("Transformer model loaded")

    def _load_baseline(self) -> None:
        """Load the baseline model (LightGBM or LogReg)."""
        from src.features.tfidf import TfidfFeaturePipeline
        from src.models.baseline import load_baseline_model

        self.tfidf_pipeline = TfidfFeaturePipeline.load(ARTIFACTS_DIR / "tfidf")
        self.label_names = self.tfidf_pipeline.label_encoder.classes_.tolist()

        # Try LightGBM first, then LogReg
        for name in ("lightgbm", "logistic_regression"):
            try:
                self.model = load_baseline_model(ARTIFACTS_DIR / "model", name)
                self.model_type = name
                logger.info("Baseline model loaded", type=name)
                return
            except FileNotFoundError:
                continue

        raise FileNotFoundError("No baseline model found in artifacts/")

    def predict(self, text: str) -> dict:
        """Classify a single document."""
        cleaned = clean_text(text)

        if self.model_type == "transformer":
            pred_label, confidence, class_probs = self.transformer.predict_single(
                cleaned
            )
        else:
            X, _ = self.tfidf_pipeline.transform([cleaned])
            proba = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            pred_label = self.label_names[pred_idx]
            confidence = float(proba[pred_idx])
            class_probs = {
                name: float(proba[i]) for i, name in enumerate(self.label_names)
            }

        # Update monitoring
        self.total_predictions += 1
        self.predictions_per_class[pred_label] += 1
        self.confidence_scores.append(confidence)

        return {
            "predicted_class": pred_label,
            "confidence": confidence,
            "probabilities": class_probs,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classify multiple documents."""
        return [self.predict(text) for text in texts]

    def get_predict_fn(self):
        """Get a prediction function suitable for LIME."""
        if self.model_type == "transformer":

            def predict_fn(texts: list[str]) -> np.ndarray:
                _, probs = self.transformer.predict(texts)
                return probs

        else:

            def predict_fn(texts: list[str]) -> np.ndarray:
                X, _ = self.tfidf_pipeline.transform(texts)
                return self.model.predict_proba(X)

        return predict_fn

    def get_metrics(self) -> dict:
        """Get current monitoring metrics."""
        avg_confidence = (
            float(np.mean(self.confidence_scores))
            if self.confidence_scores
            else 0.0
        )

        drift_detected, drift_details = self._check_drift()

        return {
            "total_predictions": self.total_predictions,
            "avg_confidence": avg_confidence,
            "predictions_per_class": dict(self.predictions_per_class),
            "drift_detected": drift_detected,
            "drift_details": drift_details,
        }

    def _load_reference_distribution(self) -> None:
        """Load reference distribution for drift detection."""
        ref_path = Path("data/reference/reference_distribution.json")
        if ref_path.exists():
            with open(ref_path) as f:
                self.reference_distribution = json.load(f)
            logger.info("Reference distribution loaded")

    def _check_drift(self) -> tuple[bool, Optional[dict]]:
        """Basic drift check comparing prediction distribution to reference."""
        if not self.reference_distribution or self.total_predictions < 50:
            return False, None

        ref_dist = self.reference_distribution.get("label_distribution", {})
        ref_total = sum(ref_dist.values())
        if ref_total == 0:
            return False, None

        ref_normalized = {k: v / ref_total for k, v in ref_dist.items()}
        pred_total = sum(self.predictions_per_class.values())
        pred_normalized = {
            k: v / pred_total for k, v in self.predictions_per_class.items()
        }

        # Simple drift: check if any class ratio shifted > 10%
        max_shift = 0.0
        shifts = {}
        for cls in ref_normalized:
            ref_ratio = ref_normalized.get(cls, 0)
            pred_ratio = pred_normalized.get(cls, 0)
            shift = abs(pred_ratio - ref_ratio)
            shifts[cls] = {"reference": ref_ratio, "current": pred_ratio, "shift": shift}
            max_shift = max(max_shift, shift)

        drift_detected = max_shift > 0.10

        return drift_detected, {
            "max_shift": max_shift,
            "class_shifts": shifts,
        }


# Singleton instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create the model service singleton."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
        _model_service.load_model()
    return _model_service
