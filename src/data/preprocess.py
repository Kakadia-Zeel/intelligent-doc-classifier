"""Text preprocessing and data splitting."""

import re

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from src.utils.config import DATA_DIR, load_config

logger = structlog.get_logger(__name__)


def load_raw_data(config_path: str = "model_config.yaml") -> pd.DataFrame:
    """Load and filter the raw complaints dataset."""
    config = load_config(config_path)
    data_config = config["data"]

    path = DATA_DIR / "raw" / "complaints.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run 'make download-data' first."
        )

    df = pd.read_csv(
        path,
        usecols=[data_config["text_column"], data_config["target_column"]],
        dtype=str,
    )

    # Drop rows without complaint narrative
    df = df.dropna(subset=[data_config["text_column"]])
    logger.info("Loaded raw data", shape=df.shape)

    # Apply label mapping to consolidate categories
    if "label_mapping" in data_config:
        df[data_config["target_column"]] = (
            df[data_config["target_column"]].replace(data_config["label_mapping"])
        )

    # Filter rare classes
    class_counts = df[data_config["target_column"]].value_counts()
    valid_classes = class_counts[
        class_counts >= data_config["min_samples_per_class"]
    ].index
    df = df[df[data_config["target_column"]].isin(valid_classes)]

    # Cap total samples for faster iteration
    max_samples = data_config.get("max_samples")
    if max_samples and len(df) > max_samples:
        df = df.sample(
            n=max_samples, random_state=data_config["random_state"]
        )

    logger.info(
        "Filtered data",
        shape=df.shape,
        n_classes=df[data_config["target_column"]].nunique(),
        classes=sorted(df[data_config["target_column"]].unique().tolist()),
    )
    return df.reset_index(drop=True)


def clean_text(text: str) -> str:
    """Clean a single text document.

    Handles CFPB-specific PII redaction patterns and normalizes text.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # CFPB redacts PII with XXXX and XX/XX/XXXX patterns
    text = re.sub(r"x{2,}", " [REDACTED] ", text)
    text = re.sub(r"xx/xx/\d{4}", " [DATE] ", text)
    text = re.sub(r"\d{2}/\d{2}/\d{4}", " [DATE] ", text)
    text = re.sub(r"\$[\d,]+\.?\d*", " [AMOUNT] ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove excessive punctuation but keep single instances
    text = re.sub(r"([!?.]){2,}", r"\1", text)

    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(
    df: pd.DataFrame, config_path: str = "model_config.yaml"
) -> pd.DataFrame:
    """Apply text preprocessing to the dataframe."""
    config = load_config(config_path)
    text_col = config["data"]["text_column"]
    prep_config = config["preprocessing"]

    df = df.copy()
    df["text_clean"] = df[text_col].apply(clean_text)

    # Filter by text length
    text_lengths = df["text_clean"].str.len()
    min_len = prep_config["min_text_length"]
    max_len = prep_config["max_text_length"] * 6  # chars, not tokens

    df = df[(text_lengths >= min_len) & (text_lengths <= max_len)]
    logger.info("Preprocessing complete", shape=df.shape)
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame, config_path: str = "model_config.yaml"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets with stratification."""
    config = load_config(config_path)
    data_config = config["data"]
    target = data_config["target_column"]

    train_val, test = train_test_split(
        df,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=df[target],
    )

    relative_val = data_config["val_size"] / (1 - data_config["test_size"])
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=data_config["random_state"],
        stratify=train_val[target],
    )

    logger.info(
        "Split complete",
        train=len(train),
        val=len(val),
        test=len(test),
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
