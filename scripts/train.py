"""CLI entrypoint for model training."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train import run_training_pipeline
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train document classification models")
    parser.add_argument(
        "--config",
        default="model_config.yaml",
        help="Model configuration file name (default: model_config.yaml)",
    )
    args = parser.parse_args()

    setup_logging()
    results = run_training_pipeline(config_path=args.config)

    print("\nTraining complete. Models saved to artifacts/")
    for name, metrics in results.items():
        print(f"  {name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
