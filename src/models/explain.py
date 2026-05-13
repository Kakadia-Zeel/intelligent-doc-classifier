"""Model explainability using LIME for text classification."""

from typing import Callable, Optional

import numpy as np
import structlog
from lime.lime_text import LimeTextExplainer

logger = structlog.get_logger(__name__)


class TextExplainer:
    """LIME-based text classification explainer."""

    def __init__(
        self,
        predict_fn: Callable[[list[str]], np.ndarray],
        class_names: list[str],
    ):
        """Initialize the explainer.

        Args:
            predict_fn: Function that takes list[str] and returns probability matrix (n_samples, n_classes)
            class_names: List of class names corresponding to prediction columns
        """
        self.predict_fn = predict_fn
        self.class_names = class_names
        self.explainer = LimeTextExplainer(
            class_names=class_names,
            split_expression=r"\W+",
            bow=True,
        )

    def explain(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 500,
        top_labels: int = 3,
    ) -> dict:
        """Generate LIME explanation for a single text.

        Returns dict with:
            - predicted_class: str
            - confidence: float
            - top_features: list of (word, weight) tuples for predicted class
            - class_probabilities: dict of class -> probability
            - all_class_features: dict of class -> list of (word, weight)
        """
        explanation = self.explainer.explain_instance(
            text,
            self.predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=top_labels,
        )

        # Get prediction
        probs = self.predict_fn([text])[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]

        # Extract features for predicted class
        top_features = explanation.as_list(label=pred_idx)

        # Features for all top labels
        all_class_features = {}
        for label_idx in explanation.available_labels():
            label_name = self.class_names[label_idx]
            all_class_features[label_name] = explanation.as_list(label=label_idx)

        result = {
            "predicted_class": pred_class,
            "confidence": float(probs[pred_idx]),
            "top_features": [
                {"word": word, "weight": float(weight)}
                for word, weight in top_features
            ],
            "class_probabilities": {
                name: float(probs[i])
                for i, name in enumerate(self.class_names)
            },
            "all_class_features": {
                cls: [
                    {"word": w, "weight": float(wt)} for w, wt in feats
                ]
                for cls, feats in all_class_features.items()
            },
        }

        logger.info(
            "Explanation generated",
            predicted_class=pred_class,
            confidence=f"{result['confidence']:.3f}",
            n_features=len(top_features),
        )
        return result

    def explain_batch(
        self,
        texts: list[str],
        num_features: int = 10,
        num_samples: int = 300,
    ) -> list[dict]:
        """Generate explanations for multiple texts."""
        return [
            self.explain(text, num_features=num_features, num_samples=num_samples)
            for text in texts
        ]
