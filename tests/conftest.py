"""Shared test fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_complaints() -> pd.DataFrame:
    """Create a small sample dataset for testing."""
    return pd.DataFrame(
        {
            "Consumer complaint narrative": [
                (
                    "I have a problem with my credit card. The bank charged me"
                    " twice for the same transaction of $500.00 on 01/15/2024."
                ),
                (
                    "My mortgage payment was not applied correctly."
                    " I sent payment on XX/XX/2024 but it shows as late."
                ),
                (
                    "The debt collector keeps calling me about a debt"
                    " that is not mine. They call XXXX times a day."
                ),
                (
                    "I applied for a student loan and was denied"
                    " without explanation. My credit score is good."
                ),
                (
                    "My checking account was overdrawn due to a bank"
                    " error. They charged me $35.00 in fees."
                ),
                (
                    "I disputed an item on my credit report but the"
                    " bureau did not investigate within 30 days."
                ),
                (
                    "The auto dealer gave me a loan with XXXX%"
                    " interest rate which seems unreasonable."
                ),
                (
                    "I received a debt collection letter for a"
                    " medical bill I already paid in full."
                ),
                (
                    "My credit card company raised my interest rate"
                    " without any notice or explanation."
                ),
                (
                    "The mortgage company is refusing to remove PMI"
                    " even though I have over 20% equity."
                ),
            ],
            "Product": [
                "Credit card",
                "Mortgage",
                "Debt collection",
                "Student loan",
                "Banking",
                "Credit reporting",
                "Vehicle loan",
                "Debt collection",
                "Credit card",
                "Mortgage",
            ],
        }
    )


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample cleaned texts for testing."""
    return [
        "i have a problem with my credit card the bank charged me twice",
        "my mortgage payment was not applied correctly",
        "the debt collector keeps calling me about a debt that is not mine",
        "i applied for a student loan and was denied without explanation",
        "my checking account was overdrawn due to a bank error",
    ]


@pytest.fixture
def sample_labels() -> list[str]:
    """Sample labels matching sample_texts."""
    return [
        "Credit card",
        "Mortgage",
        "Debt collection",
        "Student loan",
        "Banking",
    ]
