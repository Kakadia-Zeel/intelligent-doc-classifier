"""Model evaluation metrics and reporting."""

import json
from pathlib import Path

import numpy as np
import structlog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = structlog.get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """Compute comprehensive classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
    }

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    metrics["per_class"] = {
        name: {
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1-score"],
            "support": stats["support"],
        }
        for name, stats in report.items()
        if name not in ("accuracy", "macro avg", "weighted avg")
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    logger.info(
        "Evaluation complete",
        accuracy=f"{metrics['accuracy']:.4f}",
        macro_f1=f"{metrics['macro_f1']:.4f}",
        weighted_f1=f"{metrics['weighted_f1']:.4f}",
    )
    return metrics


def print_evaluation_report(metrics: dict, model_name: str = "Model") -> None:
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f" {model_name} — Evaluation Report")
    print(f"{'='*60}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
    print("\n  Per-Class Breakdown:")
    print(f"  {'Class':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>7}")
    print(f"  {'-'*50}")

    for cls_name, cls_metrics in metrics.get("per_class", {}).items():
        print(
            f"  {cls_name:<25} "
            f"{cls_metrics['precision']:>6.3f} "
            f"{cls_metrics['recall']:>6.3f} "
            f"{cls_metrics['f1']:>6.3f} "
            f"{cls_metrics['support']:>7.0f}"
        )
    print(f"{'='*60}\n")


def save_metrics(metrics: dict, path: Path, filename: str = "metrics.json") -> None:
    """Save metrics to a JSON file."""
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved", path=str(filepath))


def compare_models(results: dict[str, dict]) -> None:
    """Print a comparison table across multiple models."""
    print(f"\n{'='*70}")
    print(" Model Comparison")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Accuracy':>10} {'Macro-F1':>10} {'W-F1':>10}")
    print(f"  {'-'*60}")

    for name, metrics in results.items():
        print(
            f"  {name:<30} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['weighted_f1']:>10.4f}"
        )
    print(f"{'='*70}\n")
