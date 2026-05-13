"""Tests for text preprocessing."""

from src.data.preprocess import clean_text


class TestCleanText:
    """Tests for the clean_text function."""

    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_redact_xxxx_pattern(self):
        result = clean_text("My account XXXX1234 was charged")
        assert "[REDACTED]" in result
        assert "XXXX" not in result

    def test_redact_date_pattern_xx_format(self):
        result = clean_text("Payment on XX/XX/2024 was late")
        assert "[DATE]" in result

    def test_redact_date_pattern_numeric(self):
        result = clean_text("I paid on 01/15/2024")
        assert "[DATE]" in result
        assert "01/15/2024" not in result

    def test_redact_amount(self):
        result = clean_text("They charged me $500.00 twice")
        assert "[AMOUNT]" in result
        assert "$500" not in result

    def test_redact_large_amount(self):
        result = clean_text("The balance was $1,234,567.89")
        assert "[AMOUNT]" in result

    def test_clean_whitespace(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_preserves_meaningful_content(self):
        result = clean_text("My credit card was stolen and used fraudulently")
        assert "credit card" in result
        assert "stolen" in result
        assert "fraudulently" in result

    def test_removes_urls(self):
        result = clean_text("Visit https://example.com for details")
        assert "https://" not in result
        assert "example.com" not in result

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_collapses_repeated_punctuation(self):
        result = clean_text("This is outrageous!!!")
        assert "!!!" not in result
        assert "!" in result

    def test_combined_pii_patterns(self):
        text = "On XX/XX/2024, account XXXX was charged $500.00"
        result = clean_text(text)
        assert "[DATE]" in result
        assert "[REDACTED]" in result
        assert "[AMOUNT]" in result
