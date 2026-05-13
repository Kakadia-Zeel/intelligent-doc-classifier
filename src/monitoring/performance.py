"""Model performance tracking and alerting."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PerformanceTracker:
    """Track model performance metrics over time."""

    def __init__(self, low_confidence_threshold: float = 0.5):
        self.low_confidence_threshold = low_confidence_threshold
        self.predictions: list[dict] = []
        self.low_confidence_count = 0
        self.class_confidences: dict[str, list[float]] = defaultdict(list)

    def record_prediction(
        self, predicted_class: str, confidence: float
    ) -> Optional[dict]:
        """Record a prediction and return an alert if needed."""
        self.predictions.append(
            {"class": predicted_class, "confidence": confidence}
        )
        self.class_confidences[predicted_class].append(confidence)

        alert = None
        if confidence < self.low_confidence_threshold:
            self.low_confidence_count += 1
            alert = {
                "type": "low_confidence",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "threshold": self.low_confidence_threshold,
                "total_low_confidence": self.low_confidence_count,
            }
            logger.warning(
                "Low confidence prediction",
                predicted_class=predicted_class,
                confidence=f"{confidence:.3f}",
            )

        return alert

    def get_summary(self) -> dict:
        """Get a summary of tracked performance metrics."""
        if not self.predictions:
            return {
                "total_predictions": 0,
                "avg_confidence": 0.0,
                "low_confidence_rate": 0.0,
                "per_class_stats": {},
            }

        all_confidences = [p["confidence"] for p in self.predictions]

        per_class_stats = {}
        for cls, confs in self.class_confidences.items():
            per_class_stats[cls] = {
                "count": len(confs),
                "avg_confidence": float(np.mean(confs)),
                "min_confidence": float(np.min(confs)),
                "low_confidence_count": sum(
                    1 for c in confs if c < self.low_confidence_threshold
                ),
            }

        return {
            "total_predictions": len(self.predictions),
            "avg_confidence": float(np.mean(all_confidences)),
            "min_confidence": float(np.min(all_confidences)),
            "max_confidence": float(np.max(all_confidences)),
            "low_confidence_rate": self.low_confidence_count / len(self.predictions),
            "low_confidence_count": self.low_confidence_count,
            "per_class_stats": per_class_stats,
        }

    def save_summary(self, path: Path) -> None:
        """Save performance summary to disk."""
        path.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        with open(path / "performance_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Performance summary saved")
