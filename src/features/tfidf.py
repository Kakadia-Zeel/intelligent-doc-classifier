"""TF-IDF feature extraction pipeline."""

import pickle
from pathlib import Path

import numpy as np
import structlog
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.features.text import extract_text_features
from src.utils.config import load_config

logger = structlog.get_logger(__name__)


class TfidfFeaturePipeline:
    """Complete feature extraction pipeline: TF-IDF + text statistics."""

    def __init__(self, config_path: str = "model_config.yaml"):
        config = load_config(config_path)
        tfidf_config = config["baseline"]["tfidf"]

        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_config["max_features"],
            ngram_range=tuple(tfidf_config["ngram_range"]),
            min_df=tfidf_config["min_df"],
            max_df=tfidf_config["max_df"],
            sublinear_tf=tfidf_config["sublinear_tf"],
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
        )
        self.label_encoder = LabelEncoder()
        self._is_fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfFeaturePipeline":
        """Fit the TF-IDF vectorizer and label encoder."""
        logger.info("Fitting TF-IDF pipeline", n_docs=len(texts))
        self.vectorizer.fit(texts)
        self.label_encoder.fit(labels)
        self._is_fitted = True
        logger.info(
            "TF-IDF pipeline fitted",
            vocab_size=len(self.vectorizer.vocabulary_),
            n_classes=len(self.label_encoder.classes_),
            classes=self.label_encoder.classes_.tolist(),
        )
        return self

    def transform(
        self, texts: list[str], labels: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Transform texts into feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        # TF-IDF features
        tfidf_matrix = self.vectorizer.transform(texts)

        # Text statistics features
        text_features = extract_text_features(texts)
        text_features_array = text_features.values

        # Combine: TF-IDF (sparse) + text features (dense)
        from scipy.sparse import csr_matrix

        text_sparse = csr_matrix(text_features_array)
        features = hstack([tfidf_matrix, text_sparse])

        # Encode labels if provided
        y = None
        if labels is not None:
            y = self.label_encoder.transform(labels)

        return features, y

    def fit_transform(self, texts: list[str], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(texts, labels)
        return self.transform(texts, labels)

    def save(self, path: Path) -> None:
        """Save the fitted pipeline to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tfidf_pipeline.pkl", "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "label_encoder": self.label_encoder,
                },
                f,
            )
        logger.info("Pipeline saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "TfidfFeaturePipeline":
        """Load a fitted pipeline from disk."""
        pipeline = cls.__new__(cls)
        with open(path / "tfidf_pipeline.pkl", "rb") as f:
            data = pickle.load(f)
        pipeline.vectorizer = data["vectorizer"]
        pipeline.label_encoder = data["label_encoder"]
        pipeline._is_fitted = True
        logger.info("Pipeline loaded", path=str(path))
        return pipeline
