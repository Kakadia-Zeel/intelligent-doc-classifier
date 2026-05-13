"""Training orchestration — runs all models and logs experiments."""

import json
from pathlib import Path

import numpy as np
import structlog

from src.data.preprocess import clean_text, load_raw_data, preprocess_dataframe, split_data
from src.features.tfidf import TfidfFeaturePipeline
from src.models.baseline import save_baseline_model, train_lightgbm, train_logistic_regression
from src.models.evaluate import compare_models, compute_metrics, print_evaluation_report, save_metrics
from src.models.registry import log_experiment, setup_mlflow
from src.utils.config import load_config

logger = structlog.get_logger(__name__)

ARTIFACTS_DIR = Path("artifacts")


def run_training_pipeline(config_path: str = "model_config.yaml") -> dict:
    """Run the full training pipeline: data → features → models → evaluation."""
    config = load_config(config_path)
    training_config = load_config("training_config.yaml")

    # Setup MLflow
    setup_mlflow(
        tracking_uri=training_config["experiment"]["tracking_uri"],
        experiment_name=training_config["experiment"]["name"],
    )

    # Load and preprocess data
    logger.info("Loading data")
    df = load_raw_data(config_path)
    df = preprocess_dataframe(df, config_path)
    train_df, val_df, test_df = split_data(df, config_path)

    target_col = config["data"]["target_column"]
    text_col = "text_clean"

    train_texts = train_df[text_col].tolist()
    val_texts = val_df[text_col].tolist()
    test_texts = test_df[text_col].tolist()

    train_labels = train_df[target_col].tolist()
    val_labels = val_df[target_col].tolist()
    test_labels = test_df[target_col].tolist()

    all_results = {}

    # === TF-IDF Feature Pipeline ===
    logger.info("Building TF-IDF features")
    tfidf_pipeline = TfidfFeaturePipeline(config_path)
    X_train, y_train = tfidf_pipeline.fit_transform(train_texts, train_labels)
    X_val, y_val = tfidf_pipeline.transform(val_texts, val_labels)
    X_test, y_test = tfidf_pipeline.transform(test_texts, test_labels)

    # Save pipeline
    tfidf_pipeline.save(ARTIFACTS_DIR / "tfidf")
    label_names = tfidf_pipeline.label_encoder.classes_.tolist()

    # Save label names for serving
    (ARTIFACTS_DIR / "model").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "model" / "label_names.json", "w") as f:
        json.dump(label_names, f)

    # === Logistic Regression Baseline ===
    if training_config["training"]["run_baseline"]:
        logger.info("Training Logistic Regression baseline")
        lr_model = train_logistic_regression(X_train, y_train, config_path)
        lr_preds = lr_model.predict(X_test)
        lr_metrics = compute_metrics(y_test, lr_preds, label_names)
        print_evaluation_report(lr_metrics, "TF-IDF + Logistic Regression")

        save_baseline_model(lr_model, ARTIFACTS_DIR / "model", "logistic_regression")
        save_metrics(lr_metrics, ARTIFACTS_DIR / "metrics", "lr_metrics.json")

        log_experiment(
            model_name="TF-IDF + LogisticRegression",
            params=config["baseline"]["logistic_regression"],
            metrics=lr_metrics,
            tags={"model_type": "baseline"},
        )
        all_results["TF-IDF + LogisticRegression"] = lr_metrics

    # === LightGBM ===
    if training_config["training"]["run_lightgbm"]:
        logger.info("Training LightGBM")
        lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val, config_path)
        lgbm_preds = lgbm_model.predict(X_test)
        lgbm_metrics = compute_metrics(y_test, lgbm_preds, label_names)
        print_evaluation_report(lgbm_metrics, "TF-IDF + LightGBM")

        save_baseline_model(lgbm_model, ARTIFACTS_DIR / "model", "lightgbm")
        save_metrics(lgbm_metrics, ARTIFACTS_DIR / "metrics", "lgbm_metrics.json")

        log_experiment(
            model_name="TF-IDF + LightGBM",
            params=config["baseline"]["lightgbm"],
            metrics=lgbm_metrics,
            tags={"model_type": "lightgbm"},
        )
        all_results["TF-IDF + LightGBM"] = lgbm_metrics

    # === Transformer (DistilBERT) ===
    if training_config["training"]["run_transformer"]:
        logger.info("Training DistilBERT transformer")
        from src.models.transformer import TransformerClassifier

        transformer = TransformerClassifier(
            num_classes=len(label_names),
            label_names=label_names,
            config_path=config_path,
        )

        # Encode labels for transformer
        train_labels_enc = tfidf_pipeline.label_encoder.transform(train_labels)
        val_labels_enc = tfidf_pipeline.label_encoder.transform(val_labels)
        test_labels_enc = tfidf_pipeline.label_encoder.transform(test_labels)

        history = transformer.train(
            train_texts, train_labels_enc.tolist(),
            val_texts, val_labels_enc.tolist(),
        )

        tf_preds, tf_probs = transformer.predict(test_texts)
        tf_metrics = compute_metrics(test_labels_enc, tf_preds, label_names)
        print_evaluation_report(tf_metrics, "DistilBERT (fine-tuned)")

        transformer.save(ARTIFACTS_DIR / "transformer")
        save_metrics(tf_metrics, ARTIFACTS_DIR / "metrics", "transformer_metrics.json")

        log_experiment(
            model_name="DistilBERT (fine-tuned)",
            params=config["transformer"],
            metrics=tf_metrics,
            tags={"model_type": "transformer"},
        )
        all_results["DistilBERT (fine-tuned)"] = tf_metrics

    # === Comparison ===
    if len(all_results) > 1:
        compare_models(all_results)

    # Save reference data for monitoring
    _save_reference_distribution(test_texts, test_labels)

    logger.info("Training pipeline complete", models_trained=len(all_results))
    return all_results


def _save_reference_distribution(texts: list[str], labels: list[str]) -> None:
    """Save reference distribution for drift monitoring."""
    ref_dir = Path("data/reference")
    ref_dir.mkdir(parents=True, exist_ok=True)

    text_lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]

    from collections import Counter

    label_dist = dict(Counter(labels))

    ref_data = {
        "n_samples": len(texts),
        "text_length_mean": float(np.mean(text_lengths)),
        "text_length_std": float(np.std(text_lengths)),
        "word_count_mean": float(np.mean(word_counts)),
        "word_count_std": float(np.std(word_counts)),
        "label_distribution": label_dist,
    }

    with open(ref_dir / "reference_distribution.json", "w") as f:
        json.dump(ref_data, f, indent=2)

    logger.info("Reference distribution saved")
