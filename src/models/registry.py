"""MLflow model registry integration."""

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def setup_mlflow(tracking_uri: str = "mlruns", experiment_name: str = "doc-classifier"):
    """Set up MLflow tracking."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow configured",
        tracking_uri=tracking_uri,
        experiment=experiment_name,
    )


def log_experiment(
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts_dir: Path | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a training experiment to MLflow.

    Returns the run ID.
    """
    import mlflow

    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                mlflow.log_param(key, str(value))
            else:
                mlflow.log_param(key, value)

        # Log metrics (flatten nested dicts)
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        flat_metrics[f"{key}.{sub_key}"] = sub_value
            elif isinstance(value, (int, float)):
                flat_metrics[key] = value

        mlflow.log_metrics(flat_metrics)

        # Log artifacts
        if artifacts_dir and artifacts_dir.exists():
            mlflow.log_artifacts(str(artifacts_dir))

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        logger.info(
            "Experiment logged",
            run_id=run.info.run_id,
            model=model_name,
            accuracy=metrics.get("accuracy"),
            macro_f1=metrics.get("macro_f1"),
        )
        return run.info.run_id


def register_model(
    run_id: str,
    model_name: str = "doc-classifier",
    artifact_path: str = "model",
) -> str:
    """Register a model version in MLflow model registry.

    Returns the model version.
    """
    import mlflow

    model_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        result = mlflow.register_model(model_uri, model_name)
        logger.info(
            "Model registered",
            name=model_name,
            version=result.version,
        )
        return result.version
    except Exception as e:
        logger.warning(
            "Model registration skipped (MLflow registry may not be available)",
            error=str(e),
        )
        return "local"
