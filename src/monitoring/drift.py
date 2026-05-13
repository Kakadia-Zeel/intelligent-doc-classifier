"""Data and prediction drift detection using Evidently."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def create_text_drift_report(
    reference_texts: list[str],
    current_texts: list[str],
    reference_labels: Optional[list[str]] = None,
    current_labels: Optional[list[str]] = None,
) -> dict:
    """Generate a drift report comparing reference and current text distributions.

    Uses Evidently AI for statistical drift detection on text metadata features.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.warning("Evidently not installed, using basic drift detection")
        return _basic_drift_check(reference_texts, current_texts)

    # Build dataframes with text metadata
    ref_df = _build_text_features_df(reference_texts, reference_labels)
    cur_df = _build_text_features_df(current_texts, current_labels)

    column_mapping = ColumnMapping()
    if reference_labels is not None:
        column_mapping.target = "label"

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)

    result = report.as_dict()

    # Extract key drift metrics
    drift_summary = {
        "dataset_drift": result.get("metrics", [{}])[0]
        .get("result", {})
        .get("dataset_drift", False),
        "n_drifted_features": result.get("metrics", [{}])[0]
        .get("result", {})
        .get("number_of_drifted_columns", 0),
        "share_drifted_features": result.get("metrics", [{}])[0]
        .get("result", {})
        .get("share_of_drifted_columns", 0.0),
    }

    logger.info(
        "Drift report generated",
        dataset_drift=drift_summary["dataset_drift"],
        n_drifted=drift_summary["n_drifted_features"],
    )
    return drift_summary


def _build_text_features_df(
    texts: list[str], labels: Optional[list[str]] = None
) -> pd.DataFrame:
    """Build a DataFrame with text metadata for drift detection."""
    data = {
        "text_length": [len(t) for t in texts],
        "word_count": [len(t.split()) for t in texts],
        "avg_word_length": [
            sum(len(w) for w in t.split()) / max(len(t.split()), 1) for t in texts
        ],
        "sentence_count": [t.count(".") + t.count("!") + t.count("?") for t in texts],
        "redacted_count": [t.count("[REDACTED]") for t in texts],
        "amount_count": [t.count("[AMOUNT]") for t in texts],
        "question_ratio": [
            t.count("?") / max(len(t), 1) for t in texts
        ],
    }

    if labels is not None:
        data["label"] = labels

    return pd.DataFrame(data)


def _basic_drift_check(
    reference_texts: list[str], current_texts: list[str]
) -> dict:
    """Fallback drift detection without Evidently."""
    import numpy as np

    ref_lengths = [len(t) for t in reference_texts]
    cur_lengths = [len(t) for t in current_texts]

    ref_mean, ref_std = np.mean(ref_lengths), np.std(ref_lengths)
    cur_mean, cur_std = np.mean(cur_lengths), np.std(cur_lengths)

    # Z-test for mean shift
    z_score = abs(cur_mean - ref_mean) / max(ref_std / np.sqrt(len(cur_lengths)), 1e-6)
    drift_detected = z_score > 3.0

    return {
        "dataset_drift": drift_detected,
        "text_length_z_score": float(z_score),
        "reference_mean_length": float(ref_mean),
        "current_mean_length": float(cur_mean),
    }


def save_drift_report(report: dict, path: Path) -> None:
    """Save drift report to disk."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "drift_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Drift report saved", path=str(path))
