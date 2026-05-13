"""Text feature extraction utilities."""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


def extract_text_features(texts: Sequence[str]) -> pd.DataFrame:
    """Extract statistical features from text documents.

    These features supplement TF-IDF or transformer embeddings
    with structural/meta information about the text.
    """
    features = []
    for text in texts:
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        non_empty_sentences = [s for s in sentences if s.strip()]

        features.append(
            {
                "text_length": len(text),
                "word_count": len(words),
                "sentence_count": len(non_empty_sentences),
                "avg_word_length": (float(np.mean([len(w) for w in words])) if words else 0.0),
                "avg_sentence_length": (
                    float(np.mean([len(s.split()) for s in non_empty_sentences]))
                    if non_empty_sentences
                    else 0.0
                ),
                "redacted_count": text.count("[REDACTED]"),
                "amount_count": text.count("[AMOUNT]"),
                "date_count": text.count("[DATE]"),
                "exclamation_count": text.count("!"),
                "question_count": text.count("?"),
                "uppercase_ratio": (sum(1 for c in text if c.isupper()) / max(len(text), 1)),
                "digit_ratio": (sum(1 for c in text if c.isdigit()) / max(len(text), 1)),
                "unique_word_ratio": (len(set(words)) / max(len(words), 1)),
            }
        )
    return pd.DataFrame(features)
