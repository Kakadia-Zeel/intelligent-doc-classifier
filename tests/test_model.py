"""Model regression tests using golden dataset."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


ARTIFACTS_DIR = Path("artifacts")


class TestTfidfPipeline:
    """Tests for the TF-IDF feature pipeline."""

    def test_fit_transform(self, sample_texts, sample_labels):
        from src.features.tfidf import TfidfFeaturePipeline

        pipeline = TfidfFeaturePipeline.__new__(TfidfFeaturePipeline)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder

        pipeline.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 1))
        pipeline.label_encoder = LabelEncoder()
        pipeline._is_fitted = False

        X, y = pipeline.fit_transform(sample_texts, sample_labels)

        assert X.shape[0] == len(sample_texts)
        assert y is not None
        assert len(y) == len(sample_labels)
        assert len(np.unique(y)) == len(set(sample_labels))

    def test_transform_without_fit_raises(self, sample_texts):
        from src.features.tfidf import TfidfFeaturePipeline

        pipeline = TfidfFeaturePipeline.__new__(TfidfFeaturePipeline)
        pipeline._is_fitted = False

        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.transform(sample_texts)

    def test_label_encoding_consistency(self, sample_texts, sample_labels):
        from src.features.tfidf import TfidfFeaturePipeline

        pipeline = TfidfFeaturePipeline.__new__(TfidfFeaturePipeline)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder

        pipeline.vectorizer = TfidfVectorizer(max_features=100)
        pipeline.label_encoder = LabelEncoder()
        pipeline._is_fitted = False

        pipeline.fit(sample_texts, sample_labels)

        # Same label should always encode to same integer
        _, y1 = pipeline.transform(sample_texts[:2], sample_labels[:2])
        _, y2 = pipeline.transform(sample_texts[:2], sample_labels[:2])
        np.testing.assert_array_equal(y1, y2)


class TestEvaluation:
    """Tests for evaluation metrics computation."""

    def test_compute_metrics_perfect(self):
        from src.models.evaluate import compute_metrics

        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        labels = ["A", "B", "C"]

        metrics = compute_metrics(y_true, y_pred, labels)

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics

    def test_compute_metrics_imperfect(self):
        from src.models.evaluate import compute_metrics

        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 0, 2, 1, 1])  # 3/5 correct
        labels = ["A", "B", "C"]

        metrics = compute_metrics(y_true, y_pred, labels)

        assert 0 < metrics["accuracy"] < 1.0
        assert 0 < metrics["macro_f1"] < 1.0


class TestPerformanceTracker:
    """Tests for performance monitoring."""

    def test_record_prediction(self):
        from src.monitoring.performance import PerformanceTracker

        tracker = PerformanceTracker(low_confidence_threshold=0.5)
        alert = tracker.record_prediction("Credit card", 0.9)
        assert alert is None  # High confidence, no alert

    def test_low_confidence_alert(self):
        from src.monitoring.performance import PerformanceTracker

        tracker = PerformanceTracker(low_confidence_threshold=0.5)
        alert = tracker.record_prediction("Credit card", 0.3)
        assert alert is not None
        assert alert["type"] == "low_confidence"

    def test_summary(self):
        from src.monitoring.performance import PerformanceTracker

        tracker = PerformanceTracker()
        tracker.record_prediction("A", 0.9)
        tracker.record_prediction("B", 0.8)
        tracker.record_prediction("A", 0.7)

        summary = tracker.get_summary()
        assert summary["total_predictions"] == 3
        assert summary["per_class_stats"]["A"]["count"] == 2
        assert summary["per_class_stats"]["B"]["count"] == 1
