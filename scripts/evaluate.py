"""CLI entrypoint for model evaluation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import load_raw_data, preprocess_dataframe, split_data
from src.features.tfidf import TfidfFeaturePipeline
from src.models.baseline import load_baseline_model
from src.models.evaluate import compute_metrics, print_evaluation_report
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "lightgbm", "transformer", "all"],
        default="all",
        help="Which model to evaluate",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config("model_config.yaml")
    target_col = config["data"]["target_column"]

    # Load test data
    df = load_raw_data()
    df = preprocess_dataframe(df)
    _, _, test_df = split_data(df)

    test_texts = test_df["text_clean"].tolist()
    test_labels = test_df[target_col].tolist()

    artifacts = Path("artifacts")

    # Load TF-IDF pipeline
    tfidf_pipeline = TfidfFeaturePipeline.load(artifacts / "tfidf")
    label_names = tfidf_pipeline.label_encoder.classes_.tolist()
    X_test, y_test = tfidf_pipeline.transform(test_texts, test_labels)

    models_to_eval = []
    if args.model in ("logistic_regression", "all"):
        models_to_eval.append("logistic_regression")
    if args.model in ("lightgbm", "all"):
        models_to_eval.append("lightgbm")
    if args.model in ("transformer", "all"):
        models_to_eval.append("transformer")

    for model_name in models_to_eval:
        if model_name == "transformer":
            from src.models.transformer import TransformerClassifier

            model = TransformerClassifier.load(artifacts / "transformer")
            preds, _ = model.predict(test_texts)
            test_labels_enc = tfidf_pipeline.label_encoder.transform(test_labels)
            metrics = compute_metrics(test_labels_enc, preds, label_names)
        else:
            model = load_baseline_model(artifacts / "model", model_name)
            preds = model.predict(X_test)
            metrics = compute_metrics(y_test, preds, label_names)

        print_evaluation_report(metrics, model_name)


if __name__ == "__main__":
    main()
