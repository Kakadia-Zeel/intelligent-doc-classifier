"""Tests for text feature extraction."""

import pytest

from src.features.text import extract_text_features


class TestExtractTextFeatures:
    """Tests for the extract_text_features function."""

    def test_basic_extraction(self, sample_texts):
        features = extract_text_features(sample_texts)
        assert len(features) == len(sample_texts)
        assert "text_length" in features.columns
        assert "word_count" in features.columns

    def test_text_length(self):
        features = extract_text_features(["hello world"])
        assert features.iloc[0]["text_length"] == 11

    def test_word_count(self):
        features = extract_text_features(["one two three four"])
        assert features.iloc[0]["word_count"] == 4

    def test_sentence_count(self):
        features = extract_text_features(["First sentence. Second sentence! Third?"])
        assert features.iloc[0]["sentence_count"] == 3

    def test_redacted_count(self):
        features = extract_text_features(
            ["[REDACTED] account charged [REDACTED]"]
        )
        assert features.iloc[0]["redacted_count"] == 2

    def test_amount_count(self):
        features = extract_text_features(
            ["charged [AMOUNT] and then [AMOUNT] more"]
        )
        assert features.iloc[0]["amount_count"] == 2

    def test_empty_text(self):
        features = extract_text_features([""])
        assert features.iloc[0]["word_count"] == 0
        assert features.iloc[0]["text_length"] == 0

    def test_unique_word_ratio(self):
        # All unique
        features = extract_text_features(["one two three four"])
        assert features.iloc[0]["unique_word_ratio"] == 1.0

        # Some repeats
        features = extract_text_features(["the the the new"])
        assert features.iloc[0]["unique_word_ratio"] == 0.5

    def test_returns_all_expected_columns(self):
        features = extract_text_features(["test text"])
        expected = {
            "text_length",
            "word_count",
            "sentence_count",
            "avg_word_length",
            "avg_sentence_length",
            "redacted_count",
            "amount_count",
            "date_count",
            "exclamation_count",
            "question_count",
            "uppercase_ratio",
            "digit_ratio",
            "unique_word_ratio",
        }
        assert set(features.columns) == expected
