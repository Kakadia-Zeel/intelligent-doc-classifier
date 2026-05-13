"""Baseline models: TF-IDF + LogisticRegression and TF-IDF + LightGBM."""

import pickle
from pathlib import Path

import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression

from src.utils.config import load_config

logger = structlog.get_logger(__name__)


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config_path: str = "model_config.yaml",
) -> LogisticRegression:
    """Train a Logistic Regression baseline."""
    config = load_config(config_path)
    lr_config = config["baseline"]["logistic_regression"]

    model = LogisticRegression(
        max_iter=lr_config["max_iter"],
        C=lr_config["C"],
        class_weight=lr_config["class_weight"],
        solver="lbfgs",
        n_jobs=-1,
        random_state=config["data"]["random_state"],
    )

    logger.info("Training Logistic Regression", n_samples=x_train.shape[0])
    model.fit(x_train, y_train)
    logger.info("Logistic Regression training complete")
    return model


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    config_path: str = "model_config.yaml",
):
    """Train a LightGBM classifier."""
    import lightgbm as lgb

    config = load_config(config_path)
    lgbm_config = config["baseline"]["lightgbm"]
    n_classes = len(np.unique(y_train))

    params = {
        "objective": lgbm_config["objective"],
        "num_class": n_classes,
        "n_estimators": lgbm_config["n_estimators"],
        "learning_rate": lgbm_config["learning_rate"],
        "max_depth": lgbm_config["max_depth"],
        "num_leaves": lgbm_config["num_leaves"],
        "min_child_samples": lgbm_config["min_child_samples"],
        "subsample": lgbm_config["subsample"],
        "colsample_bytree": lgbm_config["colsample_bytree"],
        "class_weight": lgbm_config["class_weight"],
        "verbose": lgbm_config["verbose"],
        "random_state": config["data"]["random_state"],
        "n_jobs": -1,
    }

    callbacks = [lgb.log_evaluation(period=50)]
    if x_val is not None and y_val is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=lgbm_config["early_stopping_rounds"]))

    model = lgb.LGBMClassifier(**params)

    logger.info(
        "Training LightGBM",
        n_samples=x_train.shape[0],
        n_features=x_train.shape[1],
    )

    fit_params = {}
    if x_val is not None and y_val is not None:
        fit_params["eval_set"] = [(x_val, y_val)]
        fit_params["callbacks"] = callbacks

    model.fit(x_train, y_train, **fit_params)
    logger.info(
        "LightGBM training complete",
        best_iteration=getattr(model, "best_iteration_", None),
    )
    return model


def save_baseline_model(model, path: Path, name: str) -> None:
    """Save a baseline model to disk."""
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{name}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved", path=str(filepath))


def load_baseline_model(path: Path, name: str):
    """Load a baseline model from disk."""
    filepath = path / f"{name}.pkl"
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded", path=str(filepath))
    return model
